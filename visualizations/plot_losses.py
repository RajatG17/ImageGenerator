import matplotlib.pyplot as plt

def plot_losses(d_losses, g_losses, title):
    """
    Plot the losses of the GAN training process.

    Parameters:
    - d_losses: List of discriminator losses.
    - g_losses: List of generator losses.
    - title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()