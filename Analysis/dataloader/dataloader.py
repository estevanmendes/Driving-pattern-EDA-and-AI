from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataLoader:
    train_df:pd.DataFrame
    test_df:pd.DataFrame
    var_columns:list
    class_column:list
    seed=123    

    def smoothing(self,method='rolling_average',*args,**kwargs):        
        """
        method: either 'gaussian','exponential','rolling_average'
        """
        
        if method=='exponential':
            func=DataLoader.__exponential_smoothing
        elif method=='gaussian':
            func=DataLoader.__gaussian_smoothing
        else:
            func=DataLoader.__rooling_average
        
        self.train_df=self.__apply_smoothing(self.train_df,func,args,kwargs)
        self.test_df=self.__apply_smoothing(self.test_df,func,args,kwargs)
    
    def __apply_smoothing(self,df,func,func_args,func_kwargs):
        df_numeric,df_labels=df[self.var_columns],df[self.class_column]
        df_numeric=func(df_numeric,*func_args,**func_kwargs)
        return pd.concat([df_numeric,df_labels],axis=1).dropna()
    
    @staticmethod
    def __rooling_average(df:pd.DataFrame,window):
        df_smoothed=df.rolling(window).mean()
        return df_smoothed

    @staticmethod
    def __exponential_smoothing(df:pd.DataFrame,alpha):
        df_smoothed=df.ewm(alpha=alpha).mean()
        return df_smoothed
    
    @staticmethod
    def __gaussian_smoothing(df:pd.DataFrame,window,std):
        df_smoothed=df.rolling(window,win_type='gaussian',center=True).mean(std=std)
        return df_smoothed

    def __get_val_data(self):  
        np.random.seed(self.seed)                 
        X,Y=self.test_df[self.var_columns].values,self.test_df[self.class_column].values
        self.Xtest,self.Xval,self.Ytest,self.Yval=train_test_split(X,Y,test_size=.5,stratify=Y)     

    def __get_val_data_sequence(self):   
        np.random.seed(self.seed)        
        X,Y=self.XtestSequence,self.YtestSequence
        self.XtestSequence,self.XvalSequence,self.YtestSequence,self.YvalSequence=train_test_split(X,Y,test_size=.5,stratify=Y)     
    
    def __create_sequences(self,df,sequence_size):
        X=[]
        Y=[]
        for key in df[self.class_column[0]].unique():
            train_class=df[df[self.class_column[0]]==key]  
            for row_shift in range(sequence_size):
                size=(len(train_class)-row_shift)//sequence_size*sequence_size
                X_temp=train_class[self.var_columns].values[row_shift:size+row_shift]\
                                                .reshape(-1,len(self.var_columns),sequence_size)
                Y_temp=np.array([key]*len(X_temp)).reshape(-1,1)
                if len(X)!=0:
                    X=np.vstack((X,X_temp))
                    Y=np.vstack((Y,Y_temp))
                else:
                    X=X_temp
                    Y=Y_temp
        return X,Y

    def get_train_val_test_arrays(self):
        self.Xtrain=self.train_df[self.var_columns].values
        self.Ytrain=self.train_df[self.class_column].values
        self.__get_val_data()
        return self.Xtrain,self.Ytrain,self.Xval,self.Yval,self.Xtest,self.Ytest
    
    def get_train_val_test_sequence(self):
        self.__get_val_data_sequence()
        return self.XtrainSequence,self.YtrainSequence,self.XvalSequence,self.YvalSequence,self.XtestSequence,self.YtestSequence
    
    def to_sequecence(self,sequence_size):
        self.XtrainSequence,self.YtrainSequence=self.__create_sequences(self.train_df,sequence_size)
        self.XtestSequence,self.YtestSequence=self.__create_sequences(self.test_df,sequence_size)
        

    
    def to_binary_classes(self,map):
        self.train_df[self.class_column[0]]=self.train_df[self.class_column[0]].map(map)
        self.test_df[self.class_column[0]]=self.test_df[self.class_column[0]].map(map)
        
