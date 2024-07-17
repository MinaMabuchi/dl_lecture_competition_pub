import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import TransformerBrainToCLIP, train_epoch, validate, save_checkpoint_clip, load_checkpoint_clip
from src.models import ClassificationModel, save_checkpoint, load_checkpoint
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )


    # ------------------
    #     CLIP-Model
    # ------------------
    
    # model, loss, optimizer
    device = torch.device(args["device"])
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    brain_to_clip = TransformerBrainToCLIP(
        input_dim=train_set.num_channels,
        seq_len=train_set.seq_len,
        num_layers=4,
        nhead=8,
        dim_feedforward=512,
        output_dim=512
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(brain_to_clip.parameters(), lr=args.lr, weight_decay=1e-5)

    # checkpoint
    start_epoch, start_batch, train_loss, best_val_loss = load_checkpoint_clip(brain_to_clip, optimizer, args.clip_checkpoint_dir)

    # train loop
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(brain_to_clip, clip_model, train_loader, optimizer, criterion, device, epoch, start_batch)
        val_loss = validate(brain_to_clip, clip_model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.clip_model_dir, "best_brain_to_clip_model.pth")
            torch.save(brain_to_clip.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # checkpoint saving
        save_checkpoint_clip(epoch + 1, brain_to_clip, optimizer, train_loss, val_loss, 0,
                        checkpoint_dir=args.clip_checkpoint_dir,
                        filename=f"brain_to_clip_checkpoint_epoch_{epoch+1}.pt")

        # reset batch
        start_batch = 0

    print("BrainToCLIP training completed.")


    # -----------------------------
    #     Calssification Model
    # -----------------------------
    device = torch.device(args.device)

    # BrainToCLIPmodel load
    brain_to_clip = TransformerBrainToCLIP(
        input_dim=train_set.num_channels,
        seq_len=train_set.seq_len,
        num_layers=4,
        nhead=8,
        dim_feedforward=512,
        output_dim=512
    ).to(device)
    # checkpoint load
    checkpoint = torch.load(args.brain_to_clip_path)
    # only dict load
    brain_to_clip.load_state_dict(checkpoint['model_state_dict'])
    brain_to_clip.eval()  # evaluation mode

    # freezing params
    for param in brain_to_clip.parameters():
        param.requires_grad = False

    
    
    # model, loss, optimizer
    classification_model = ClassificationModel(
        num_classes=train_set.num_classes,
        seq_len=train_set.seq_len,
        in_channels=train_set.num_channels,
        clip_dim=512,  # BrainToCLIP
        num_subjects=len(torch.unique(train_set.subject_idxs)),
        hid_dim=128,
        embedding_dim=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Accuracy reset
    accuracy_metric = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(device)

    # checkpoint load
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists(os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "latest_checkpoint.pt"))
        classification_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        classification_model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training")):
            meg_data, labels, subject_idxs, images = [b.to(device) for b in batch]
            meg_data_reshaped = meg_data.view(meg_data.size(0), meg_data.size(1), -1)  # (batch_size, channels, time_steps)
            #print(f"Original MEG data shape: {meg_data.shape}")
            #print(f"Reshaped MEG data shape: {meg_data_reshaped.shape}")

            with torch.no_grad():
                clip_features = brain_to_clip(meg_data_reshaped)

            #print(f"CLIP features shape: {clip_features.shape}")
            outputs = classification_model(meg_data, clip_features, subject_idxs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy_metric(outputs, labels).item()

            if (batch_idx + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # validation loop
        classification_model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                meg_data, labels, subject_idxs, images = [b.to(device) for b in batch]
                meg_data_reshaped = meg_data.view(meg_data.size(0), meg_data.size(1), -1)

                clip_features = brain_to_clip(meg_data_reshaped)
                outputs = classification_model(meg_data, clip_features, subject_idxs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += accuracy_metric(outputs, labels).item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # result printing
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classification_model.state_dict(), os.path.join(args.model_dir, "best_model.pt"))
            print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")

        # checkpoint saving
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': classification_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }, os.path.join(args.checkpoint_dir, "latest_checkpoint.pt"))
        
    print("Training completed.")

    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------


    # Load the best model
    model_path = os.path.join(args.model_dir, "best_model.pt")
    classification_model.load_state_dict(torch.load(model_path))


    # ------------------
    #  Start evaluation
    # ------------------
    preds = []
    classification_model.eval()
    for batch in tqdm(test_loader, desc="Validation"):
        meg_data, subject_idxs = [b.to(device) for b in batch]
        meg_data_reshaped = meg_data.view(meg_data.size(0), meg_data.size(1), -1)

        with torch.no_grad():
            clip_features = brain_to_clip(meg_data_reshaped)
            outputs = classification_model(meg_data, clip_features, subject_idxs)
            preds.append(outputs.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission_21"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()
