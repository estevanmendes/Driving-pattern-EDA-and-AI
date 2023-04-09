
from keras_tuner import RandomSearch

class random_search_cv(RandomSearch):

    def __init__(self,model,parameters,cv=None,*args,**kwargs):
        self.model_class=model
        self.parameters=parameters
        self.cv=cv        
        if 'directory' not in kwargs.keys():
            kwargs['directory']= str(self) 
        if 'project_name' not in kwargs.keys():
            kwargs['project_name']=str(model)+str(datetime.datetime.now()).replace('-','').replace('.','').replace(' ','').replace(':','') 
        super().__init__(*args,**kwargs)

        
    def run_trial(self, trial, **kwargs):        
        hp = trial.hyperparameters
        parameters=self.process_parameters(hp,self.parameters)
        return self.model_class.model_tunable(cv=self.cv,**parameters)

    
    def best_hyperparams_display(self):
       self.results_summary(1)

    def display_multiobjective_results(self,objetives):
        result=""
        for objective in objetives:
            result+=objective+" : "+str(self.oracle.get_best_trials()[0].metrics.metrics[objetives[0]].get_history())+"\n"
        print(result)
    

    @staticmethod
    def process_parameters(hp,kwargs)->dict:
        new_kwargs={}
        for k,v in kwargs.items():
            if 'choice' in v:
                new_kwargs[k]=RandomSearch._choice(hp,(k,v))
            elif 'int' in v:
               new_kwargs[k]=RandomSearch._int(hp,(k,v))
            elif'random' in v:
                new_kwargs[k]=RandomSearch._random((k,v))

        return new_kwargs    
    
    @staticmethod
    def _random(parameters):
        k,v=parameters
        index=np.random.randint(len(v['random']))
        return v['random'][index]
    
    @staticmethod
    def _choice(hp,parameter):
        return hp.Choice(name=parameter[0],**parameter[1]['choice'])
        
    
    @staticmethod
    def _int(hp,parameter):        
        return hp.Int(name=parameter[0],**parameter[1]['int'])


    def __str__(self) -> str:
        return 'RandomSearch'