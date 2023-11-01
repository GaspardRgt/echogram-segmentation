
import torch


from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               out: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float]:
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_rfcm_loss = 0
  train_ce_loss = 0
  train_total_loss = 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      y_logits = model(X)
      y_pred = out(y_logits)

      rfcm_loss, ce_loss, total_loss = loss_fn(y_pred, X, y)
      
      train_rfcm_loss += rfcm_loss.item()
      train_ce_loss += ce_loss.item()
      train_total_loss += total_loss.item() 

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

  # Adjust metrics to get average loss and accuracy per batch 
  train_rfcm_loss /= len(dataloader)
  train_ce_loss /= len(dataloader)
  train_total_loss /= len(dataloader)
  
  return train_rfcm_loss, train_ce_loss, train_total_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              out: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  test_rfcm_loss = 0
  test_ce_loss = 0
  test_total_loss = 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      y_logits = model(X)
      y_pred = out(y_logits)

      rfcm_loss, ce_loss, total_loss = loss_fn(y_pred, X, y)
      
      test_rfcm_loss += rfcm_loss.item()
      test_ce_loss += ce_loss.item()
      test_total_loss += total_loss.item() 

  # Adjust metrics to get average loss and accuracy per batch 
  test_rfcm_loss /= len(dataloader)
  test_ce_loss /= len(dataloader)
  test_total_loss /= len(dataloader)
  
  return test_rfcm_loss, test_ce_loss, test_total_loss


def train(model: torch.nn.Module, 
          out: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader, 
          valid_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer:torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
   
    # Create empty results dictionary
    results = {"train_rfcm_loss": [],
               "train_ce_loss": [],
               "train_total_loss": [],
               "test_rfcm_loss": [],
               "test_ce_loss": [], 
               "test_total_loss": []
    }
    
    writer = SummaryWriter()

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_rfcm_loss, train_ce_loss, train_total_loss = train_step(model=model,
                                                                      dataloader=train_dataloader,
                                                                      loss_fn=loss_fn,
                                                                      out=out,
                                                                      optimizer=optimizer,
                                                                      device=device)
        
        test_rfcm_loss, test_ce_loss, test_total_loss = test_step(model=model,
                                                                  dataloader=valid_dataloader,
                                                                  loss_fn=loss_fn,
                                                                  out=out,
                                                                  device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_total_loss:.4f} | "
          f"test_loss: {test_total_loss:.4f}"
        )

        # Update results dictionary
        results["train_rfcm_loss"].append(train_rfcm_loss)
        results["train_ce_loss"].append(train_ce_loss)
        results["train_total_loss"].append(train_total_loss)
        results["test_rfcm_loss"].append(test_rfcm_loss)
        results["test_ce_loss"].append(test_ce_loss)
        results["test_total_loss"].append(test_total_loss)

        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Total loss", 
                           tag_scalar_dict={"train_total_loss": train_total_loss,
                                            "test_total_loss": test_total_loss},
                           global_step=epoch)
        
        writer.add_scalars(main_tag="RFCM loss", 
                           tag_scalar_dict={"train_rfcm_loss": train_rfcm_loss,
                                            "test_rfcm_loss": test_rfcm_loss},
                           global_step=epoch)
        
        writer.add_scalars(main_tag="CE loss", 
                           tag_scalar_dict={"train_ce_loss": train_ce_loss,
                                            "test_ce_loss": test_ce_loss},
                           global_step=epoch)
        
        # Track the PyTorch model architecture
        writer.add_graph(model=model, 
                         # Pass in an example input
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))
    
    # Close the writer
    writer.close()

    # Return the filled results at the end of the epochs
    return results