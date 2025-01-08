from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

[train_dataset, valloader] = torch.load('wikiloader_ssquote_ssbord')
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)


device='cuda'
sent = SentenceTransformer("all-mpnet-base-v2", device=device)

class Model(nn.Module):
    def __init__(self, d_emb, original_model):
        super().__init__()
        hidden_size = original_model.config.hidden_size
        
        self.emb = nn.Linear(d_emb, hidden_size)
        self.emb_rev = nn.Linear(hidden_size, d_emb)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.mod = get_peft_model(original_model, lora_config)

    def forward(self, x):
        x = self.emb(x)
        
        x = self.mod(inputs_embeds=x)
        x = x.hidden_states[-1]
        
        x = self.emb_rev(x)
        return x

#orig_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_hidden_states=True)
#orig_model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
orig_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", output_hidden_states=True)

model = Model(sent.get_sentence_embedding_dimension(), orig_model).to(device)


optimizer = Adam([
    *model.emb.parameters(),
    *model.emb_rev.parameters(),
    *model.mod.parameters(),
])

pad = 99

def criterion(output, target):
    mask = target == pad
    output = output[~mask]
    target = target[~mask]
    mse = nn.MSELoss(reduction='none'); out = mse(output, target).sum()
    #mse = nn.MSELoss(); out = mse(output, target)
    #cossim = nn.CosineSimilarity(dim=-1, eps=1e-6); out = 1-cossim(output, target) #PEUT-ETRE DIM=1
    return out

val_list=[]

def test(epoch):
    model.eval()
    with torch.no_grad():
        src = next(iter(valloader))
        outputs = model(src[:, :-1])
        val_loss = criterion(outputs, src[:, 1:]).item()
        print('Epoch', str(epoch+1) + ', Loss:', val_loss)
        val_list.append(val_loss)

test(-1)

for epoch in range(1000):
    model.train()
    for src in trainloader:
        optimizer.zero_grad()
        outputs = model(src[:, :-1])
        loss = criterion(outputs, src[:, 1:])
        loss.backward()
        optimizer.step()
        #scheduler.step()

    test(epoch)
    
    #early stopping code
    if val_list[-1] < min(val_list[:-1]):#its performs better, we save the model
        model_save = model
    elif (len(val_list) - val_list.index(min(val_list))) > 3: #no better model in the last epochs
        break;

model = model_save
plt.plot(val_list)
plt.show()
