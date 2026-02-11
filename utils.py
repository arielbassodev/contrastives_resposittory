import matplotlib.pyplot as plt

def plot_training_loss(epoch_losses, epochs):
 plt.figure(figsize=(8, 6))
 plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Training Loss')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.title('Training Loss per Epoch')
 plt.grid(True)
 plt.legend()
 plt.show()
    