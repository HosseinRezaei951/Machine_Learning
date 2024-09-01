import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


for x in range(5):

    size = [3,5,10]

    for s in size:

        # trian
        x_train=np.sort(np.array(random.sample(range(1, 100), s)))
        x_train=x_train[:,np.newaxis]
        x_train=x_train/100
        y_train=-0.3+0.5*np.sin(x_train*np.pi*2)

        # test
        x_test=np.sort(np.array(random.sample(range(1, 100), 99)))
        x_test=x_test[:,np.newaxis]
        x_test=x_test/100
        y_test=-0.3+0.5*np.sin(x_test*np.pi*2)

        fig, axs = plt.subplots(1, 3,figsize=(32, 9))

        one_arr_trian = np.ones((len(x_train),1))
        one_arr_test = np.ones((len(x_test),1))
         
        alpha=0.001
        beta=240
        

        ####################################################
        #### Phi_3 
        ####################################################
        tmp=alpha*np.eye(4)
        phi_train3=np.concatenate((one_arr_trian,x_train,np.power(x_train,2),np.power(x_train,3)),axis=1)
      
        # Make predictions
        S_N_i3=tmp+beta*np.matmul(np.transpose(phi_train3),phi_train3)
        m_N3=np.matmul(np.matmul(beta*np.linalg.inv(S_N_i3),np.transpose(phi_train3)),y_train)
        phi_test3=np.concatenate((one_arr_test,x_test,np.power(x_test,2),np.power(x_test,3)),axis=1)

        y_pred_mean3 = np.transpose(np.matmul(np.transpose(m_N3),np.transpose(phi_test3)))
        y_pred_std3 = np.diag(1/beta+np.matmul(np.matmul(phi_test3,np.linalg.inv(S_N_i3)),np.transpose(phi_test3)))
        y_pred_std3 =np.transpose(y_pred_std3)


        axs[0].scatter(x_train, y_train,  color='black', label="Train")
        axs[0].plot(x_test, y_test, color='green', linewidth=3, label="Test")
        axs[0].plot(x_test, y_pred_mean3, color='blue', linewidth=3, label="Mean Prediction")
        axs[0].plot(x_test, np.transpose(np.transpose(y_pred_mean3)+3*y_pred_std3), color='red', linewidth=3, label="Uncertainty Interval")
        axs[0].plot(x_test, np.transpose(np.transpose(y_pred_mean3)-3*y_pred_std3), color='red', linewidth=3)
        axs[0].set_title('Phi_3')
        axs[0].legend()
        


        ####################################################
        #### Phi_5 
        ####################################################
        tmp=alpha*np.eye(6)
        phi_train5=np.concatenate((one_arr_trian,x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5)),axis=1)
      
        # Make predictions
        S_N_i5=tmp+beta*np.matmul(np.transpose(phi_train5),phi_train5)
        m_N5=np.matmul(np.matmul(beta*np.linalg.inv(S_N_i5),np.transpose(phi_train5)),y_train)
        phi_test5=np.concatenate((one_arr_test,x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5)),axis=1)

        y_pred_mean5 = np.transpose(np.matmul(np.transpose(m_N5),np.transpose(phi_test5)))
        y_pred_std5 = np.diag(1/beta+np.matmul(np.matmul(phi_test5,np.linalg.inv(S_N_i5)),np.transpose(phi_test5)))
        y_pred_std5 =np.transpose(y_pred_std5)

        axs[1].scatter(x_train, y_train,  color='black', label="Train")
        axs[1].plot(x_test, y_test, color='green', linewidth=3, label="Test")
        axs[1].plot(x_test, y_pred_mean5, color='blue', linewidth=3, label="Mean Prediction")
        axs[1].plot(x_test, np.transpose(np.transpose(y_pred_mean5)+3*y_pred_std5), color='red', linewidth=3, label="Uncertainty Interval")
        axs[1].plot(x_test, np.transpose(np.transpose(y_pred_mean5)-3*y_pred_std5), color='red', linewidth=3)
        axs[1].set_title('Phi_5')
        axs[1].legend()


        ####################################################
        #### Phi_10 
        ####################################################
        tmp=alpha*np.eye(11)
        phi_train10=np.concatenate((one_arr_trian,x_train,np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7),np.power(x_train,8),np.power(x_train,9),np.power(x_train,10)),axis=1)
      
        # Make predictions
        S_N_i10=tmp+beta*np.matmul(np.transpose(phi_train10),phi_train10)
        m_N10=np.matmul(np.matmul(beta*np.linalg.inv(S_N_i10),np.transpose(phi_train10)),y_train)
        phi_test10=np.concatenate((one_arr_test,x_test,np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7),np.power(x_test,8),np.power(x_test,9),np.power(x_test,10)),axis=1)

        y_pred_mean10 = np.transpose(np.matmul(np.transpose(m_N10),np.transpose(phi_test10)))
        y_pred_std10 = np.diag(1/beta+np.matmul(np.matmul(phi_test10,np.linalg.inv(S_N_i10)),np.transpose(phi_test10)))
        y_pred_std10 =np.transpose(y_pred_std10)


        axs[2].scatter(x_train, y_train,  color='black', label="Train")
        axs[2].plot(x_test, y_test, color='green', linewidth=3, label="Test")
        axs[2].plot(x_test, y_pred_mean10, color='blue', linewidth=3, label="Mean Prediction")
        axs[2].plot(x_test, np.transpose(np.transpose(y_pred_mean10)+3*y_pred_std10), color='red', linewidth=3, label="Uncertainty Interval")
        axs[2].plot(x_test, np.transpose(np.transpose(y_pred_mean10)-3*y_pred_std10), color='red', linewidth=3)
        axs[2].set_title('Phi_10')
        axs[2].legend()


        ####################################################
        #### Save Plots
        ####################################################
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig("Results/"+str(s)+"_"+str(x+1)+".jpeg")