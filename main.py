import torch
from dataset import ImageNet100Dataset
from torch.utils.data import DataLoader
from config import batch_size, device, LR, beta1, beta2, weight_decay, Num_gen, Validation_batches, train_split, checkpoint
from net import Net
from tqdm import tqdm

"""def _image_to_patches_(self, image, patch_size):
        batch_size, num_channels, image_height, image_width = image.shape
        unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)
        patches = unfold(image).transpose(1, 2).reshape(-1, num_channels, patch_size, patch_size)

        extra_patch = torch.ones(1, num_channels, patch_size, patch_size, device=image.device)
        patches = torch.cat([extra_patch, patches], dim=0)
        patches = patches.view(batch_size, -1, patch_size * patch_size * num_channels)

        return patches"""

def validitaion(model, val_loader, loss_fn):
    model.eval()
      # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    loss = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (data, targets) in enumerate(val_loader):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)  # Forward pass
            _, predicted = torch.max(outputs, 1)
            #_, targets = targets.max(dim=1)
            
            loss += loss_fn(outputs, targets).item()

            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            if batch_idx == Validation_batches:  # Validation on 10 batches
                break
        
    accuracy = 100 * correct_predictions / total_samples
    print(f'Validation Accuracy: {accuracy:.2f}%')

    return loss / Validation_batches, accuracy

def main():
    train_dataset = ImageNet100Dataset(root_dir = "C:/Users/bucan/Desktop/Imagenet", json_file= "C:/Users/bucan/Desktop/Imagenet/Labels.json", split = "train")
    val_dataset =  ImageNet100Dataset(root_dir = "C:/Users/bucan/Desktop/Imagenet", json_file= "C:/Users/bucan/Desktop/Imagenet/Labels.json", split="val")
    print(len(train_dataset))
    print(len(val_dataset))
    train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
    val_loader = DataLoader(dataset= val_dataset, batch_size= batch_size, shuffle= True)
    model = Net().to(device)
    if checkpoint:
        param = torch.load("ViT_param.pth")
        model.load_state_dict(param)
        print("Checkpoint loaded")
    
    optim = torch.optim.Adam(model.parameters(), lr = LR, betas = (beta1, beta2), weight_decay= weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()

    stop_traning = 0
    best_loss = 100
    best_acc = 0
    
    for gen in range(Num_gen):
        if stop_traning < 40:
            loss_sum = 0
            for i, (img, label) in enumerate(train_loader):
                if stop_traning < 40:
                    with torch.cuda.amp.autocast():
                        img = img.to(device)  # Move the input image to the device
                        label = label.to(device) 
                         
                        pred = model(img)

                        loss = loss_fn(pred, label)
                        
                        loss.backward()

                        optim.step()

                        optim.zero_grad()

                    loss_sum += loss.item()

                    if (i)%100 == 0:
                        print("After", i+1, "bacthes: ", loss.item())
                    
                    if i % 200 == 0:
                        val_loss, acc = validitaion(model, val_loader, loss_fn)
                        print(f"Validation loss after {i} ittereons in {gen} epoch: {val_loss:.2f}")
                        best_acc = max(best_acc, acc)
                        if val_loss < best_loss:
                            stop_traning = 0
                            best_loss = val_loss
                            model.save("ViT_param.pth")
                        else:
                            stop_traning += 1

                        model.train()

                else:
                    break
            print("Generation:", gen, "----Loss:", loss_sum / len(train_loader))
            #validate(model, loader)
            model.train()
        else:
            break
    print(f"Najbolji postignuti val loss je: {best_loss:.2f}") 
    print(f"Najbolji accuracy {best_acc:.2f}")   

if __name__ == "__main__":
    main()