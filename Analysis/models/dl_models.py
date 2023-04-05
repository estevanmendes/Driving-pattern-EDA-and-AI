from abc import abstractclassmethod
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import classification_report,accuracy_score,ConfusionMatrixDisplay,confusion_matrix,precision_score,recall_score,roc_curve,roc_auc_score,balanced_accuracy_score
import matplotlib.pyplot as plt

class DLModel():

    def __init__(self,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest):
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.Xval=Xval
        self.Yval=Yval
        self.Xtest=Xtest
        self.Ytest=Ytest
        self.training_history=[]
        self.model=self.model_compiler()
    
    @property
    def architecture(self):
        return self.model.summary()
    
    @abstractclassmethod
    def model_compiler(self):
        pass

    @abstractclassmethod
    def train_params(self)->dict:
        pass   

    def train(self,seed=123,verbose=0):
        np.random.seed(seed)
        history=self.model.fit(self.Xtrain,self.Ytrain,validation_data=(self.Xval,self.Yval),verbose=verbose,**self.training_params())
        self.training_history.append(history)     

    def test(self):
        threshold=0.5
        self.predictions=self.model.predict(self.Xtest)
        self.binary_predictions=(self.predictions>threshold).astype(int)
        test_accuracy=accuracy_score(self.Ytest,self.binary_predictions)        
        test_balanced_accuracy=balanced_accuracy_score(self.Ytest,self.binary_predictions)        

        test_precision=precision_score(self.Ytest,self.binary_predictions)
        test_recall=recall_score(self.Ytest,self.binary_predictions)
        auc=roc_auc_score(self.Ytest,self.binary_predictions)
        print(f'Test accuracy : {test_accuracy*100:.0f} %')
        print(f'Test balanced accuracy : {test_balanced_accuracy*100:.0f} %')

        print(f'Test precision : {test_precision*100:.0f} %')
        print(f'Test recall : {test_recall*100:.0f} %')
        print(f'Model AUC : {auc*100:.0f} %')
        self.confusion_matrix_plot()


    @staticmethod
    def __unpack_data_history(history_record,key):
        key=DLModel.__get_metric_key_by_prefix(key,history_record)
        metric=[]
        for train_data in history_record:
            metric.extend(train_data.history[key])
        
        return metric
    
    @staticmethod
    def __get_metric_key_by_prefix(key,history_record):
        key_in_history=''
        if 'val' in key:
            for metric in history_record[-1].history.keys():
                if key in metric:
                    key_in_history=metric
                    break
        else:
            for metric in history_record[-1].history.keys():
                if key in metric and 'val' not in metric:
                    key_in_history=metric
                    break

        return key_in_history

    def loss_plot(self,ax=None):
        if not ax:
            loss=DLModel.__unpack_data_history(self.training_history,'loss')
            loss_val=DLModel.__unpack_data_history(self.training_history,'val_loss')

            plt.title('loss')
            plt.plot(loss,label='train')
            plt.plot(loss_val,label='val')

            plt.xlabel('epochs')
            plt.show()
        else:
            pass

    def accuracy_plot(self,ax=None):
        if not ax:        
            accuracy=DLModel.__unpack_data_history(self.training_history,'binary_accuracy')
            val_accuracy=DLModel.__unpack_data_history(self.training_history,'val_binary_accuracy')
            plt.title('Accuracy')
            plt.plot(accuracy,label='train')
            plt.plot(val_accuracy,label='val')
            plt.legend()
            plt.xlabel('epochs')
            plt.grid(True)
            plt.show()
        else:
            pass


    def precision_recall_plot(self,ax=None):
        if not ax:
            precision=DLModel.__unpack_data_history(self.training_history,'precision')
            recall=DLModel.__unpack_data_history(self.training_history,'recall')
            val_precision=DLModel.__unpack_data_history(self.training_history,'val_precision')
            val_recall=DLModel.__unpack_data_history(self.training_history,'val_recall')
            plt.subplot(1,2,1)
            plt.title('Precision')
            plt.plot(precision,label='Train')
            plt.plot(val_precision,label='Val') 
            plt.xlabel('epochs')
            plt.legend()
            plt.subplot(1,2,2)
            plt.title('Recall')
            plt.plot(recall,label='Train')
            plt.plot(val_recall,label='Val') 
            plt.xlabel('epochs')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            pass
    
    
    def confusion_matrix_plot(self):
        if hasattr(self, 'predictions'):
            cm = confusion_matrix(self.Ytest,self.binary_predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
        else:
            print('Test method should be called before')

    def plot_all_training_metrics(self):
        self.loss_plot()
        self.accuracy_plot()
        self.precision_recall_plot()

    def multiple_seeds_analysis(self,seeds:list):
        pass
          
    
