

DATASET_DIR = '/users/data/d3m/datasets/training_datasets'
METADATA_PATH = '~/data/closed_d3m_unembed_data.csv'
embed_col_path = ''
embed_all_path = ''
MAX_LEN = 20
MAX_CELLS = 100

def parse_datasets(datasets):
    raw_data, header, groups = [], [], []
    for dataset_id, dataset_doc_path in datasets.items():
        # open the dataset doc to get the column headers
        with open(dataset_doc_path, 'r') as dataset_doc:
            meta_dataset = json.load(dataset_doc)
            for resource in meta_dataset['dataResources']:
                if 'columns' not in resource:
                    continue
                # then open the actual dataset table to get column values
                if resource['resPath'][-4:] == '.csv':
                    try:
                        dataset = pd.read_csv(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath']))
                    except:
                        continue
                else:
                    values = 0
                    tables = []
                    for entry in os.scandir(os.path.join(os.path.dirname(dataset_doc_path), resource['resPath'])):
                        tables.append(pd.read_csv(entry.path))
                        values += len(tables[-1])
                        if values >= MAX_CELLS:
                            break
                    dataset = pd.concat(tables, ignore_index=True)

                for column in resource['columns']:
                    values = list(dataset[column['colName']].values)
                    if len(values) > MAX_CELLS:
                        values = [str(v)[:MAX_LEN] for v in values[:MAX_CELLS]]
                    else:
                        values = [str(v)[:MAX_LEN] for v in values] + ['' for i in range(MAX_CELLS - len(values))]

                    raw_data.append(values)
                    header.append((column['colType'],))
                    groups.append(meta_dataset['about']['datasetName'])
                    
    return np.asarray(raw_data), np.asarray(header), np.asarray(groups)

#write function to get data into tables
##########################################



def initialize_model(, weights_model_path):
    model = SentenceTransformer(model_path)
    return model

def embed_data(data, weights_model_path, col_name):
    index = data.index.to_list()
    model = initialize_model(model_weights_path)
    if (col_name is True):
        col_name_embs = model.encode(df['colName'].to_numpy())
    else:
        dataset_name_embs = model.encode(df['datasetName'].to_numpy())
        description_embs = model.encode(df['description'].to_numpy())
        col_name_embs = model.encode(df['colName'].to_numpy())

    del model
    #it needs to preserve indeces of passed in data
    embeddings_df = pd.DataFrame(data=np.hstack((dataset_name_embs, description_embs, col_name_embs)), columns=['emb_{}'.format(i) for i in range(3*len(col_name_embs[0]))],index=index)
    #delete to conserve memory
    del index
    del dataset_name_embs
    del description_embs
    del col_name_embs
    return embeddings_df
    
def save_results(results,conf):
    #save the results to a csv file
    results.to_csv(model_name+'_final_cross_val.csv',index=False)
    filename = model_name+'_matrix_mean.pkl'
    fileObject = open(filename, 'wb')
    pickle.dump(conf, fileObject)
    fileObject.close()
    
def evaluate_model():
    #this first part needs to grab the data and embed it
    if (rank == 0):
        if (use_metadata is True):
            #build the table of metadata from the directory - todo
            #####################################
            #get the relavant info from the table
            col_types = data['colTypes']
            dataset_names = data['datasetName']
            splits = np.array_split(data_df.drop(['colTypes'],axis=1), COMM.size)
        else:
            #build the table and get the data for use in SIMON
            #########################################
    else:
        splits = None  
    splits = COMM.scatter(splits, root=0)
    results_embed = embed_data(splits, col_name=col_name)
    results_embed = MPI.COMM_WORLD.gather(results_init, root = 0)
    del splits
    #gather the embedding results
    if (rank == 0):
        X_data = pd.concat(results_embed,axis=0)
        del results_embed
        y = column_types
        groups = dataset_names
    #stop the processes until this is done    
    comm.Barrier()
    
    
    #this next part will take the embedded data and run cross validation on it
    if (rank == 0):
        k_splitter = LeaveOneGroupOut()
        #split_num = k_splitter.get_n_splits(X_data, groups=groups)
        jobs = list(k_splitter.split(X_data, groups=groups))
        #gets the jobs and splits them into even-sized-lists to be spread across the different cpu's
        list_jobs_total = [list() for i in range(COMM.size)]
        for i in range(len(jobs)):
            j = i % COMM.size
            list_jobs_total[j].append(jobs[i])
        jobs = list_jobs_total

    else:
        #initalizes variables to pass to other processors, size of X_data must be intialized correctly
        if (use_metadata):
            if (col_name is True):
                X_data = np.empty((47831, 768),dtype='d')
            else:
                print("bad")
                X_data = np.empty((47831, 768*3), dtype='d')
        else:
            X_data = np.empty((30592, 100, 20), dtype='d')
        jobs = None
        y = None
    
    y = COMM.bcast(y,root=0)
    jobs = COMM.scatter(jobs, root=0)
    COMM.Bcast([X_data, MPI.FLOAT], root=0)
    
    #run cross-validation on all the different processors
    results_init = []
    for job in jobs:
        train_ind, test_ind = job
        results_init.append(run_fold(train_ind, test_ind, balance=balance))

    #gather results together
    results_init = MPI.COMM_WORLD.gather(results_init, root = 0)
    del jobs

    #compile and save the results
    if (rank == 0):
        del X_data
        del y
        print("Finished cross validation!")
        #flatten the total results
        results_final = [_i for temp in results_init for _i in temp]
        y_test = list()
        y_hat = list()
        #compute the results
        for hat, test in results_final:
            y_test += test
            y_hat += hat
        accuracy = accuracy_score(y_test, y_hat)
        f1_macro = f1_score(y_test, y_hat, average='macro')
        f1_micro = f1_score(y_test, y_hat, average='micro')
        f1_weighted = f1_score(y_test, y_hat, average='weighted')
        conf = confusion_matrix(y_test, y_hat)
        save_results(results, conf) 
        return results

    return results





















   

