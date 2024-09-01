import matplotlib.pyplot as plt
import numpy as np
import random

for x in range(5):

    size = [5,10,20]

    for i in size:
        x_train=np.sort(np.array(random.sample(range(1, 100), i)))
        x_train=x_train[:,np.newaxis]
        x_train=x_train/100
        y_train=-0.3+0.5*np.sin(x_train*np.pi*2)

        x_test=np.sort(np.array(random.sample(range(1, 100), 99)))
        x_test=x_test[:,np.newaxis]
        x_test=x_test/100
        y_test=-0.3+0.5*np.sin(x_test*np.pi*2)
        landa=0.001

        fig, axs = plt.subplots(3, 3,figsize=(32, 18))
    

        ####################################################
        #### Phi_1 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[0, 0].scatter(x_test, y_test, color='blue', label="Test")
        axs[0, 0].scatter(x_train, y_train,  color='red', label="Train")
        axs[0, 0].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[0, 0].set_title('Phi_1')
        axs[0, 0].legend()

        


        ####################################################
        #### Phi_2 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[0, 1].scatter(x_test, y_test, color='blue', label="Test")
        axs[0, 1].scatter(x_train, y_train,  color='red', label="Train")
        axs[0, 1].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[0, 1].set_title('Phi_2')
        axs[0, 1].legend()




        ####################################################
        #### Phi_3 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[0, 2].scatter(x_test, y_test, color='blue', label="Test")
        axs[0, 2].scatter(x_train, y_train,  color='red', label="Train")
        axs[0, 2].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[0, 2].set_title('Phi_3')
        axs[0, 2].legend()




        ####################################################
        #### Phi_4 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[1, 0].scatter(x_test, y_test, color='blue', label="Test")
        axs[1, 0].scatter(x_train, y_train,  color='red', label="Train")
        axs[1, 0].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[1, 0].set_title('Phi_4')
        axs[1, 0].legend()




        ####################################################
        #### Phi_5 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[1, 1].scatter(x_test, y_test, color='blue', label="Test")
        axs[1, 1].scatter(x_train, y_train,  color='red', label="Train")
        axs[1, 1].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[1, 1].set_title('Phi_5')
        axs[1, 1].legend()




        ####################################################
        #### Phi_6 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[1, 2].scatter(x_test, y_test, color='blue', label="Test")
        axs[1, 2].scatter(x_train, y_train,  color='red', label="Train")
        axs[1, 2].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[1, 2].set_title('Phi_6')
        axs[1, 2].legend()




        ####################################################
        #### Phi_7 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[2, 0].scatter(x_test, y_test, color='blue', label="Test")
        axs[2, 0].scatter(x_train, y_train,  color='red', label="Train")
        axs[2, 0].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[2, 0].set_title('Phi_7')
        axs[2, 0].legend()




        ####################################################
        #### Phi_8 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7),np.power(x_train,8)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7),np.power(x_test,8)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[2, 1].scatter(x_test, y_test, color='blue', label="Test")
        axs[2, 1].scatter(x_train, y_train,  color='red', label="Train")
        axs[2, 1].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[2, 1].set_title('Phi_8')
        axs[2, 1].legend()




        ####################################################
        #### Phi_9 
        ####################################################
        # Train Phase
        phi_train=np.concatenate((np.ones((len(x_train),1)),x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7),np.power(x_train,8),np.power(x_train,9)),axis=1)
        weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi_train),phi_train)+landa*np.eye(np.shape(phi_train)[-1])),
                            np.transpose(phi_train)),y_train)	

        # Test Phase
        phi_test=np.concatenate((np.ones((len(x_test),1)),x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7),np.power(x_test,8),np.power(x_test,9)),axis=1)
        y_pred = np.matmul(phi_test,weight)


        # Plot outputs
        axs[2, 2].scatter(x_test, y_test, color='blue', label="Test")
        axs[2, 2].scatter(x_train, y_train,  color='red', label="Train")
        axs[2, 2].plot(x_test, y_pred, color='green', linewidth=3, label="Prediction Line")
        axs[2, 2].set_title('Phi_9')
        axs[2, 2].legend()




        ####################################################
        #### Save Plot
        ####################################################
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig("Results/"+str(x+1)+"_"+str(i)+".jpeg")
