Autoencoder using fully connected neural network (FCNN)

Author: T M Feroz Ali


Network architecture:

    Autoencoder and Decoder uses 3 layer n/w.
    
    Latent space dimn: 9 
    
    All RelU non-linearity except Sigmoid for output layer
    


Other details:

    Loss: BCELoss

    Analyzes latent space using PCA

    Dataset: MNIST




Plot of train and test losses:

![alt text](https://github.com/ferozalitm/AutoEncoder_FCNN/blob/main/Results/Loss.png)



Reconstructrion on training-set samples:

![alt text](https://github.com/ferozalitm/AutoEncoder_FCNN/blob/main/Results/train_reconst-150.png)



Reconstructrion on test-set samples:

![alt text](https://github.com/ferozalitm/AutoEncoder_FCNN/blob/main/Results/test_reconst-150.png)



Analysis of VAE latent space using PCA:

![alt text](https://github.com/ferozalitm/AutoEncoder_FCNN/blob/main/Results/PCA_latentSpace-145.png)

