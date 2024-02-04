import pickle
import torch
from helper import tokenize, forward_ab, f1_score, accuracy, precision, recall
import pandas as pd
import random
from tqdm import tqdm
import os
from models import CrossEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import re
from transformers import AutoTokenizer
import torch
from cosine_sim import CrossEncoder_cossim
import torch.nn.functional as F
##These methods are used for pruning 
# Mapping words to numbers for comparison
number_mapping = {
    "ten": 10, "twenty": 20, "thirty": 30, 
    "forty": 40, "fifty": 50
}

def normalize_expression(expr):
    # Split the expression into sub-expressions by commas, if any, for separate processing
    sub_expressions = expr.split(',')
    # Normalize each sub-expression and track if it's a simple assignment
    normalized_info = [(normalize_sub_expression(sub.strip()), 'simple' in sub.strip()) for sub in sub_expressions]
    # Conditionally sort the normalized sub-expressions if they are not simple assignments
    # Only sort if none of the sub-expressions are simple assignments or if their order is unchanged
    if all(not is_simple for _, is_simple in normalized_info):
        normalized_info.sort(key=lambda x: x[0])  # Sort based on the normalized expression
    return ', '.join([ni[0] for ni in normalized_info])

def normalize_sub_expression(sub_expr):
    # Identify all components (words and numbers) and operators
    components = re.findall(r'\w+|[=!<>]+', sub_expr)
    if len(components) == 3 and components[1] in ['=', '!=']:  # Simple equalities or inequalities
        # Sort the two elements for these cases, but mark as simple assignment if number is on the right
        if components[0].isdigit() or components[2].isdigit():
            # If a number is involved, it's a simple assignment, don't sort
            pass
        elif components[0] > components[2]:
            components[0], components[2] = components[2], components[0]
    elif len(components) > 3 and components[1] in ['=', '!=']:  # Complex expressions with operations
        # Sort elements on the right side of the expression if it's a complex expression
        if '+' in sub_expr:
            # Split the right side further by '+' and sort
            right_side = sorted(sub_expr.split(components[1])[1].replace(' ', '').split('+'))
            # Reassemble the expression with the sorted right side
            components = [components[0], components[1]] + ['+'.join(right_side)]
    return ' '.join(components)

def extract_colors_and_numbers(text):
    colors = ["red", "blue", "green", "yellow", "purple"]
    numbers = list(number_mapping.keys())
    found_elements = {"colors": [], "numbers": []}
    for color in colors:
        if color in text:
            found_elements["colors"].append(color)
    for number in numbers:
        if number in text:
            found_elements["numbers"].append(number_mapping[number])
    return found_elements


def is_valid_common_ground(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
    #print(cg_colors, cg_numbers)
    color_match = not elements["colors"] or set(cg_colors) == set(elements["colors"])
    number_match = not elements["numbers"] or set(cg_numbers) == set(elements["numbers"])
    return color_match and number_match

def is_valid_individual_match(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]

    for color in elements["colors"]:
        for number in elements["numbers"]:
            if color in cg_colors and number in cg_numbers:
                return True
    return False
    

def extract_colors_and_numbers(text):
    colors = ["red", "blue", "green", "yellow", "purple"]
    numbers = list(number_mapping.keys())
    found_elements = {"colors": [], "numbers": []}
    for color in colors:
        if color in text:
            found_elements["colors"].append(color)
    for number in numbers:
        if number in text:
            found_elements["numbers"].append(number_mapping[number])
    return found_elements


def is_valid_common_ground(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
    color_match = not elements["colors"] or set(cg_colors) == set(elements["colors"])
    number_match = not elements["numbers"] or set(cg_numbers) == set(elements["numbers"])
    return color_match and number_match

def is_valid_individual_match(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
    for color in elements["colors"]:
        for number in elements["numbers"]:
            if color in cg_colors and number in cg_numbers:
                return True
    return False
    

def make_proposition_map(dataset):
    data = f'./Data/{dataset}.csv'
    df = pd.read_csv(data)
    prop_dict = defaultdict(dict)
    #normalise the common ground 
    #print(df['Common Ground'])
    df['Common Ground'] = df['Common Ground'].apply(lambda x: normalize_expression(x.replace("and", ",")))
    df['Common Ground']
    for x, y in enumerate(df.iterrows()):

        prop_dict[x]['common_ground'] = df['Common Ground'][x]
        prop_dict[x]['transcript'] = df['Transcript'][x]
        prop_dict[x]['label'] = df['Label'][x]
        prop_dict[x]['group'] = df['Group'][x]
    return prop_dict, df
def test_make_proposition_map(dataset):
    
    df = dataset
    prop_dict = defaultdict(dict)
    for x, y in enumerate(df.iterrows()):

        prop_dict[x]['common_ground'] = df['Common Ground'][x]
        prop_dict[x]['transcript'] = df['Transcript'][x]
    return prop_dict, df

def add_special_tokens(proposition_map):
    for x, y in proposition_map.items():
        #print(y['common_ground'])
        cg_with_token = "<m>" + " " + y['common_ground']+ " "  + "</m>"
        prop_with_token = "<m>" + " "+ y['transcript'] +" " + "</m>"
        proposition_map[x]['common_ground'] = cg_with_token
        proposition_map[x]['transcript'] = prop_with_token
    return proposition_map

def predict_with_XE(parallel_model, dev_ab, dev_ba, device, batch_size, cosine_sim=False):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    description='Predicting'
    if(cosine_sim):
        description = 'Getting Cosine'
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_indices = indices[i: i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices,cosine_sim=False)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices,cosine_sim=False)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba) 


def train_prop_XE(dataset, model_name=None,n_splits=10):
    dataset_folder = f'./datasets/{dataset}/'
    device = torch.device('cuda:0')
    device_ids = list(range(1))
    #load the statement and proposition data
    prop_dict, df = make_proposition_map("Dataset_Updated")
    proposition_map = add_special_tokens(prop_dict)
    
    #train_pairs  = [x for x in proposition_map.keys()]
    #train_labels = [y['label'] for x,y in proposition_map.items()]

    #dev_pairs = [x for x in proposition_map.keys()]
    #dev_labels = [y['label'] for x,y in proposition_map.items()]
    groups = df['Group'].values

    # Setting up group k-fold cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    #train(train_pairs, train_labels, dev_pairs, dev_labels, parallel_model, proposition_map, dataset_folder, device,
        #    batch_size=20, n_iters=10, lr_lm=0.000001, lr_class=0.0001)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        print(f"Training on fold {fold+1}")

        train_pairs = train_idx.tolist()  # Convert numpy array to list
        train_labels = [df['Label'].iloc[idx] for idx in train_idx]
        dev_pairs = test_idx.tolist()  # Convert numpy array to list
        dev_labels = [df['Label'].iloc[idx] for idx in test_idx]
        group = df['Group'].iloc[test_idx[0]]
        '''
        #try a sample unit test with a smaller pos/neg set
        train_pairs =train_pairs[0:50] + train_pairs[-50:] # 50 pos and 50 neg labels
        train_labels = train_labels[0:50] + train_labels[-50:]
        dev_pairs = dev_pairs[0:50] + dev_pairs[-50:]  # 50 pos and 50 neg labels
        dev_labels = dev_labels[0:50] + dev_labels[-50:]
        '''

        #print(train_pairs)
        #print('Training - ',train_pairs, len(train_pairs))
        #print('Testing = ', dev_pairs,len(dev_pairs))
        model_name = 'roberta-base'
        scorer_module = CrossEncoder(is_training=True,long=False,  model_name=model_name).to(device)

        parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
        parallel_model.module.to(device)
        #Only using pos pairs
        #train(train_pairs, train_labels, dev_pairs, dev_labels, parallel_model, proposition_map, dataset_folder, device,
        #    batch_size=20, n_iters=10, lr_lm=0.000001, lr_class=0.0001)
        
        train(train_pairs, train_labels, dev_pairs, dev_labels, parallel_model, proposition_map, dataset_folder, device,
            batch_size=20, n_iters=10, lr_lm=0.000001, lr_class=0.0001,group =group)
        break
        
 
  
def tokenize_props(tokenizer, proposition_ids, proposition_map, m_end, max_sentence_len=1024, truncate=True):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for index in proposition_ids:
        sentence_a = proposition_map[index]['transcript']
        sentence_b = proposition_map[index]['common_ground']

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                   ' '.join([doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        return tokenized_ab_dict

    if truncate:
        tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
        tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)
    else:
        instances_ab = [' '.join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [' '.join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(list(instances_ab), add_special_tokens=False, padding=True)

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab['input_ids'])

        tokenized_ab = {'input_ids': torch.LongTensor(tokenized_ab['input_ids']),
                         'attention_mask': torch.LongTensor(tokenized_ab['attention_mask']),
                         'position_ids': torch.arange(tokenized_ab_input_ids.shape[-1]).expand(tokenized_ab_input_ids.shape)}

        tokenized_ba = tokenizer(list(instances_ba), add_special_tokens=False, padding=True)
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba['input_ids'])
        tokenized_ba = {'input_ids': torch.LongTensor(tokenized_ba['input_ids']),
                        'attention_mask': torch.LongTensor(tokenized_ba['attention_mask']),
                        'position_ids': torch.arange(tokenized_ba_input_ids.shape[-1]).expand(tokenized_ba_input_ids.shape)}

    return tokenized_ab, tokenized_ba    
    

# Tokenize the test_transcripts here, similarly to how you did for train and dev sets
# You can use tokenize_props or a similar function, depending on how you need the data to be structured for testing

    
def train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          proposition_map,
          working_folder,
          device,
          batch_size=16,
          n_iters=50,
          lr_lm=0.00001,
          lr_class=0.001,
          group=20):
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])


    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba = tokenize_props(tokenizer, train_pairs, proposition_map, parallel_model.module.end_id, max_sentence_len=512)
    dev_ab, dev_ba = tokenize_props(tokenizer, dev_pairs, proposition_map, parallel_model.module.end_id, max_sentence_len=512)
   
    #labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    print("train tensor size",train_ab['input_ids'].size())
    print("dev tensor size",dev_ab['input_ids'].size())
    print("train label size", len(train_labels))
    print("dev label size", len(dev_labels))
    train_loss = []
    #print('This is the pairs - ', train_ab)
    for n in range(n_iters):
        
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices, cosine_sim=False)
            scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices, cosine_sim=False)

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(scores_mean, batch_labels)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        train_loss.append(iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_scores_ab, dev_scores_ba = predict_with_XE(parallel_model, dev_ab, dev_ba, device, batch_size,cosine_sim=False)
        dev_predictions = (dev_scores_ab + dev_scores_ba)/2
        print(dev_predictions)
        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)
        print(dev_predictions)
        print(dev_predictions, dev_labels)
        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev recall:", recall(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))
        plt.plot(train_loss)
        plt.show()
     
        # if n % 2 == 0:
        #     scorer_folder = working_folder + f'/XE_scorer/chk_{n}'
        #     if not os.path.exists(scorer_folder):
        #         os.makedirs(scorer_folder)
        #     model_path = scorer_folder + '/linear.chkpt'
        #     torch.save(parallel_model.module.linear.state_dict(), model_path)
        #     parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        #     parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
        #     print(f'saved model at {n}')

    
    #This creates the test dataset with only the positive pairs 
    def create_test_set(dev_pairs, dev_labels, proposition_map):
        positive_dev_pairs = [pair for pair, label in zip(dev_pairs, dev_labels) if label == 1]
        
        test_instances = []
        for pair in positive_dev_pairs:
            transcript = proposition_map[pair]['transcript'].replace("<m>", "").replace("</m>", "").strip()
            common_ground = proposition_map[pair]['common_ground']
            test_instances.append({'transcript': transcript, 'common_ground': common_ground})

        return test_instances

    
    test_instances = create_test_set(dev_pairs, dev_labels, proposition_map) 

    # Create a DataFrame from the test instances
    test_df = pd.DataFrame(test_instances, columns=['transcript', 'common_ground'])
    test_df["Label"] = 1
    #get the list of all possible common grounds
    common_grounds_dataSet = pd.read_csv('/s/babbage/b/nobackup/nblancha/public-datasets/ilideep/XE/Data/OracleWithLabels/props/correctedList.csv')
    common_grounds = list(common_grounds_dataSet['Propositions'])
    
    new_rows = []
    
    parallel_model = parallel_model.to(device)
    evaluation_results = []
    genericCosine = False
    #for each of the transctipt in the test dataset, get the transcript and generate the pruned possible common grounds. 
    for index, row in test_df.iterrows():
        elements = extract_colors_and_numbers(row['transcript'].lower()) #The list of colors / weights in the transcript
        filtered_common_grounds = []
        filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)]

        if not filtered_common_grounds:  # If no match found, try individual color-number pairs
            filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]  #If there is no match where only the mentioned colors and weights are present, get the individual combincations 
        filtered_common_grounds = [normalize_expression(expr) for expr in filtered_common_grounds] #normalize filtered
    
        if(len(filtered_common_grounds)==1650 or len(filtered_common_grounds)==1):
            continue
        #This is where the cosine pruning will happen. filtered_common_grounds is the list of all the pruned possible common grounds  
        #We will now get the cosine similarity for row['transcript'] and those pruned (filtered common grounds) 
        #Save the top 5 cosine pairs
        
        cosine_similarities = []
        for cg in filtered_common_grounds:
            
            if(genericCosine):
                input = parallel_model.module.tokenizer.encode_plus(row['transcript'].lower(), cg, add_special_tokens=True, return_tensors="pt")
                input_ids = input['input_ids'].to(device)
                attention_mask = input['attention_mask'].to(device)
                position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(device)

                # Generate vector representations
                _, transcript_vec, common_ground_vec = parallel_model.module.generate_cls_arg_vectors(
                    input_ids, attention_mask, position_ids, None, None, None
                )

                # Calculate cosine similarity
                cosine_similarity = F.cosine_similarity(transcript_vec, common_ground_vec).item()
                cosine_similarities.append(cosine_similarity)
            else:
            # Tokenize and prepare inputs using the tokenizer from parallel_model
                cg_with_token = "<m>" + " " + cg + " "  + "</m>"
                trans_with_token = "<m>" + " "+ row['transcript'] +" " + "</m>"
                theIndividualDict = {
                    "transcript": trans_with_token,
                    "common_ground": cg_with_token # match[0] is the common ground text
                }
                #new_df = pd.DataFrame(theIndividualDict, columns=["transcript", "common_ground"])
                #indecies = new_df.index.to_list()
                proposition_map = {0: theIndividualDict} 
                proposition_ids = [0]
                tokenizer = parallel_model.module.tokenizer
                test_ab, test_ba = tokenize_props(tokenizer,proposition_ids,proposition_map,parallel_model.module.end_id ,max_sentence_len=512, truncate=True)    
                
                
                
                cosine_test_scores_ab, cosine_test_scores_ba = predict_with_XE(parallel_model, test_ab, test_ba, device, batch_size,cosine_sim=True)
                cosine_similarity = cosine_test_scores_ab + cosine_test_scores_ba /2
                cosine_similarities.append(cosine_similarity)
            

        # Select top 5 matches based on cosine similarity
        top_matches = sorted(zip(filtered_common_grounds, cosine_similarities), key=lambda x: x[1], reverse=True)[:5]
        if not top_matches:  # If top_matches is empty
            print(f"Transcript: {row['transcript'].lower()}")
            print("Filtered common grounds with no top matches:", filtered_common_grounds)
            break
        
        #print("transcript - ", row['transcript'].lower())
        #print("common_ground - ", row['common_ground'])    
        #print("top matches - " , top_matches) 
        
         # For each top match, create a new row with the transcript and the common ground
        for match in top_matches:
            new_row = {
                "transcript": row['transcript'],
                "common_ground": match[0]  # match[0] is the common ground text
            }
            new_rows.append(new_row)
    
    
    new_df = pd.DataFrame(new_rows, columns=["transcript", "common_ground"])
    new_df.index.to_list()#the list of indicies in the dict that needs to be tokenized
    
    proposition_map_test = new_df.to_dict(orient='index') #make it into a dict
    proposition_map_test = add_special_tokens(proposition_map_test)    # add the special tokens to transcript and common ground 

    #call tokenize props here.
    tokenizer = parallel_model.module.tokenizer
    new_df.to_csv("test_set.csv") #sanity check
    
    
    test_ab, test_ba = tokenize_props(tokenizer, new_df.index.to_list(), proposition_map_test, parallel_model.module.end_id, max_sentence_len=512, truncate=True)    
    
    test_scores_ab, test_scores_ba = predict_with_XE(parallel_model, test_ab, test_ba, device, batch_size,cosine_sim=False)
    test_predictions = (test_scores_ab + test_scores_ba)/2
    new_df["scores"] = test_predictions #Get the raw scores as given by the cross Encoder
    test_predictions = test_predictions > 0.5
    test_predictions = torch.squeeze(test_predictions)
    #new_df["scores"] = test_predictions
    highest_scoring_pairs = new_df.loc[new_df.groupby('transcript')['scores'].idxmax()]
    
    # Step 3: Verify against the correct common grounds
    # Assuming 'test_df' has a unique row for each transcript with the correct common ground
    actual_common_ground_map = test_df.set_index('transcript')['common_ground'].to_dict()
    new_df['actual_common_ground'] = new_df['transcript'].map(actual_common_ground_map)# Set transcript as index for easy lookup
    new_df['Group'] =  group
    new_df.to_csv(f'resultsTrainedCosine/{group}.csv')

    correct_matches = 0
    # for index, row in highest_scoring_pairs.iterrows():
    #     transcript = row['transcript']
    #     print('transcript - ', transcript)
    #     predicted_common_ground = row['common_ground']
    #     #actual_common_ground = test_df.loc[transcript, 'common_ground']
    #     actual_common_ground = test_df.loc[test_df['transcript'] == transcript, 'common_ground'].values[0]
    
    #     actual_common_ground = actual_common_ground.replace("<m>", "").replace("</m>", "").replace(" ", "")
    #     print('the actual common ground -', actual_common_ground)
    #     print('predectied -', predicted_common_ground)
    #     if predicted_common_ground == actual_common_ground:
    #         correct_matches += 1
    #         print("got a match")
    # final_accuracy = correct_matches / len(highest_scoring_pairs)
    # print(f"Accuracy of correctly identifying the true common ground: {final_accuracy:.2f}")
    # print(new_df)
    #ask abhijnan: I need to get the top common ground. I am only getting a label of 1 or 0. Where do i access the actual sentences/ common grond 
    #print(test_predictions, test_labels)
    #print("dev accuracy:", accuracy(dev_predictions, dev_labels))
    #print("dev precision:", precision(dev_predictions, dev_labels))
    #print("dev recall:", recall(dev_predictions, dev_labels))
    #print("dev f1:", f1_score(dev_predictions, dev_labels))
"""

    #Predict here. Create the dataset. Prune with Heuristic. Prune with cosine. use predict_with_XE
    #get all the positive labels from dev set 
    #get the pruning done. This will have over 200 possible pairs for each positive label
    #get the top cosine similarity of the top 5. 
    # predict 

    # scorer_folder = working_folder + '/XE_scorer/'
    # if not os.path.exists(scorer_folder):
    #     os.makedirs(scorer_folder)
    # model_path = scorer_folder + '/linear.chkpt'
    # torch.save(parallel_model.module.linear.state_dict(), model_path)
    # parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    # parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
"""

if __name__ == '__main__':
    train_prop_XE('ecb', model_name='roberta-base')

