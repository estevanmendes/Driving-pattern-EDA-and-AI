from dataclasses import dataclass
from pyts.image import RecurrencePlot
from dataloader import DataLoader

@dataclass
class RecurrenceImageLoader(DataLoader):

    
    def ordenate_data(self,columns_order:list,images_depth:list):
        self.train_df=self.train_df[columns_order]
        self.test_df=self.test_df[columns_order]
        self.images_depth=images_depth
    

    def recurrence_plot_params(self,**kwargs):
        self.RecurrencePloter=RecurrencePlot(**kwargs)


    def generate_recurrence_plots(self):
        self.XtrainRPImgs=self.__recurrence_plots(self.XtrainSequence)
        self.XvalRPImgs=self.__recurrence_plots(self.XvalSequence)
        self.XtestRPImgs=self.__recurrence_plots(self.XtestSequence)
    
    def __recurrence_plots(self,Tensor)-> np.array:
        X_img_channels=[]
        for Matrix in Tensor:
                img_channels_temp=[]    
                for Vec in Matrix:
                  img_channels_temp.append(np.squeeze(self.RecurrencePloter.transform(Vec.reshape(-1,len(Vec))),axis=0))
                X_img_channels.append(img_channels_temp)

        return np.swapaxes(np.array(X_img_channels),1,-1)
    
    
    def customize_recurrance_plots(self):
        self.XtrainRPImgs=self.__customize_recurrence_plot(self.XtrainRPImgs)
        self.XvalRPImgs=self.__customize_recurrence_plot(self.XvalRPImgs)
        self.XtestRPImgs=self.__customize_recurrence_plot(self.XtestRPImgs)

    def __customize_recurrence_plot(self,imgs):      
        if hasattr(self,'images_depth'):
            depths=self.images_depth
        else:
            raise Exception("The method ordenate_data should be called before")
        new_imgs=[]
        for img in imgs:
            channels=0
            new_img_temp=[]
            for depth in depths:     
                new_img_temp.append(self.concat_channels(*np.swapaxes(img[:,:,range(channels,channels+depth)],-1,0)))
                channels+=depth
            new_imgs.append(self.concat_imgs(*new_img_temp))

        return np.array(new_imgs)


    def expand_image_size(self,new_shape):
        """ new_shape is a iterable in each the first item sets how height will be expanded and the second item will sets the width expansion """
        self.XtrainRPImgs=self._expand_image_by_replicating(self.XtrainRPImgs,new_shape)
        self.XvalRPImgs=self._expand_image_by_replicating(self.XvalRPImgs,new_shape)
        self.XtestRPImgs=self._expand_image_by_replicating(self.XtestRPImgs,new_shape)
        return self.XtrainRPImgs,self.XvalRPImgs,self.XtestRPImgs

    @staticmethod
    def _expand_image_by_replicating(tensor,new_shape:list)->np.array:
        """ 
            For each channel replicates n times some line, and m times some collumn in order to expand them 
        """
        new_tensor_shape=np.ones(4)
        new_tensor_shape[1:3]=new_shape
        new_tensor=np.zeros(shape=np.multiply(tensor.shape,new_tensor_shape).astype(int))
        for index,matrix in enumerate(tensor):
            new_tensor[index]=np.repeat(matrix,new_shape[0],axis=0).repeat(new_shape[1],axis=1)
            
        return new_tensor



    @staticmethod
    def concat_channels(*args):
        rgb_img=np.stack((args),axis=-1)
        return rgb_img
    
    @staticmethod
    def concat_imgs(*args):
        concated_rgb_img=np.vstack((args))
        return concated_rgb_img
    
    def get_train_val_test_recurrence_img(self):
        if not hasattr(self,'XvalSequence'):
            self.get_train_val_test_sequence()
        self.generate_recurrence_plots()
        return self.XtrainRPImgs,self.YtrainSequence,self.XvalRPImgs,self.YvalSequence,self.XtestRPImgs,self.YtestSequence

    def get_train_val_test_recurrence_img_customized(self):
        if not hasattr(self,'XvalSequence'):
            self.get_train_val_test_sequence()
        self.generate_recurrence_plots()
        self.customize_recurrance_plots()
        return self.XtrainRPImgs,self.YtrainSequence,self.XvalRPImgs,self.YvalSequence,self.XtestRPImgs,self.YtestSequence



        
                    
                    