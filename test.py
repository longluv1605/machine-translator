def train_model(
    model, data_loader, optimizer, criterion, scaler, clip, device, epochs=1
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["en_ids"].to(device)
        trg = batch["vi_ids"].to(device)
        src_mask, trg_mask = generate_masks(src, trg)

        # Forward
        lr = get_lr(global_step, model_dim=512, warmup_steps=4000)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast("cuda"):
            output = model(src, trg, device)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)

        # Update
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Batch: {i + 1}/ {len(data_loader)}: Loss {epoch_loss / (i+1):.3f}")
        if (i + 1) % 3000 == 0:
            torch.save(model.state_dict(), "/content/drive/MyDrive/model.pth")
        del src, trg, output, loss, batch
        torch.cuda.empty_cache()
    return epoch_loss / len(data_loader)
