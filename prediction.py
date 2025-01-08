from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from sentence_transformers import SentenceTransformer, util
device='cuda'
sent = SentenceTransformer("all-mpnet-base-v2", device=device)

import pandas as pd
max_len = 20


def predict(transfo, src, steps=max_len):
    if steps > max_len:
        print('steps could not be superior to max_len (who is '+str(max_len)+')')
        print('steps is set to '+str(max_len))
        steps = max_len
    if steps == 0:
        print('steps is set to 1')
        steps = 1

    transfo.eval()
    with torch.no_grad():

        text = sent.encode(src, convert_to_tensor=True).to("cuda").unsqueeze(0)
        
        while text.size(1) < steps:
            output = (transfo(text))[:,-1:]
            text = torch.cat((text, output), dim=1)

        similarities = util.semantic_search(text[0, 1:], corpus_embedding) #delete the batch and the first token
        affichage = [[f"{round(sim['score'], 2)} {corpus[sim['corpus_id']]}" for sim in liste] for liste in similarities] #score, corpus dans un string


        for aff in src:
            print(aff, '\n')
        for aff in affichage:
            print(aff[0])
        
        df = pd.DataFrame(affichage).transpose()
        df.columns = [i+len(src) for i in df.columns]
        
        for i, source in enumerate(src):
            new_column = [source] + [""] * (len(df) - 1)
            df.insert(i, str(i), new_column)

        return df


#src = [' = John Lenon = \n']
#src = ['John Lenon']
src = ['= John Lenon =', 'John Winston Ono Lennon(born John Winston Lennon; 9 October 1940 - 8 December 1980) was an English singer-songwriter, musician and political activist. He gained worldwide fame as the founder, co-lead vocalist and rhythm guitarist of the Beatles. His work included music, writing, drawings and film. His songwriting partnership with Paul McCartney remains the most successful in history as the primary songwriters in the Beatles.']
#src = ['Chocolate cake recipe']
#src = ['= Chocolate cake =', 'Chocolate cake or chocolate gâteau (from French: gâteau au chocolat) is a cake flavored with melted chocolate, cocoa powder, or both. It can also have other ingredients such as fudge, vanilla creme, and other sweeteners.']
#src = ['import pandas as pd']
predict(model, src)