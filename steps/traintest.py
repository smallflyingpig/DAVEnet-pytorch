import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .util import *
from dataloaders.image_caption_dataset import sort_data

import tqdm
from tensorboardX import SummaryWriter

def train(audio_model, image_model, train_loader, test_loader, args):
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    writer = SummaryWriter(log_dir=exp_dir)

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    if args.cuda and (not isinstance(audio_model, torch.nn.DataParallel)):
        audio_model = nn.DataParallel(audio_model)

    if args.cuda and (not isinstance(image_model, torch.nn.DataParallel)):
        image_model = nn.DataParallel(image_model)

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    image_model.train()
    global_cnt = len(train_loader)*epoch
    recalls = validate(audio_model, image_model, test_loader, args)
    while True:
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        loader_bar = tqdm.tqdm(train_loader, ncols=100)
        for i, data in enumerate(loader_bar):
            global_cnt += 1
            # measure data loading time
            data_time_this = time.time() - end_time
            data_time.update(data_time_this)
            data = sort_data(data, key_idx=2, descending=True)
            (image_input, audio_input, nframes) = data

            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            image_input = image_input.to(device)

            optimizer.zero_grad()

            audio_output, sent_emb = audio_model(audio_input, nframes=nframes)
            image_output, image_emb = image_model(image_input)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.div_(pooling_ratio)

            #if args.fast_flag:
            #    loss = sampled_margin_rank_loss_fast(image_output, audio_output,
            #        nframes, margin=args.margin, simtype=args.simtype)
            #else:
            #    loss = sampled_margin_rank_loss(image_output, audio_output,
            #        nframes, margin=args.margin, simtype=args.simtype)

            loss = jointly_margin_rank_loss_fast(image_emb, sent_emb, margin=args.margin)
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time_this = time.time() - end_time
            batch_time.update(batch_time_this)
            writer.add_scalar(tag="loss", scalar_value=loss, global_step=global_cnt)

            loader_bar.set_description('Epoch: [{0}]\t'
                  'Batch {batch_time:.3f}\t'
                  'Data {data_time:.3f}\t'
                  'Loss {loss:.4f}'.format(
                   epoch, batch_time=batch_time_this,
                   data_time=data_time_this, loss=loss.cpu().item()))
            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('\nEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        loader_bar.close()
        
        recalls = validate(audio_model, image_model, test_loader, args)
        
        avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
        writer.add_scalar(tag="accu", scalar_value=avg_acc, global_step=global_cnt)

        torch.save(audio_model.state_dict(),
                "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        torch.save(image_model.state_dict(),
                "%s/models/image_model.%d.pth" % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
        
        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc
            shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_audio_model.pth" % (exp_dir))
            shutil.copyfile("%s/models/image_model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_image_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1


def validate(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if args.cuda else "cpu")
    batch_time = AverageMeter()
    if args.cuda and (not isinstance(audio_model, torch.nn.DataParallel)):
        audio_model = nn.DataParallel(audio_model)
    if args.cuda and (not isinstance(image_model, torch.nn.DataParallel)):
        image_model = nn.DataParallel(image_model)
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = sort_data(data, key_idx=2, descending=True)
            (image_input, audio_input, nframes) = data
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            
            # compute output
            image_output, image_emb = image_model(image_input)
            audio_output, sent_emb = audio_model(audio_input, nframes)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            # I_embeddings.append(image_output)
            # A_embeddings.append(audio_output)

            I_embeddings.append(image_emb)
            A_embeddings.append(sent_emb)
            
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.div_(pooling_ratio)

            frame_counts.append(nframes.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        nframes = torch.cat(frame_counts)

        #recalls = calc_recalls(image_output, audio_output, nframes, simtype=args.simtype, fast_flag=args.fast_flag)
        recalls = calc_recalls_emb(image_output, audio_output)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
    print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)

    return recalls
