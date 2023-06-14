import os
import sys
from torch import nn
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_process import  *

device = "cuda" if torch.cuda.is_available() else "cpu"
#加载模型
net,_= clip.load('ViT-B/16', 'cpu')
net = net.to(device)
#自注意力层
from torch.cuda.amp import autocast, GradScaler


class attention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        '''
        :param embed_dim: 嵌入特征个数
        :param num_heads: scale dot-product attention层数
        '''
        super(attention, self).__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.w_q=[nn.Linear(embed_dim,embed_dim).to(device) for i in range(num_heads)]
        self.w_k=[nn.Linear(embed_dim,embed_dim).to(device) for i in range(num_heads)]
        self.w_v=[nn.Linear(embed_dim,embed_dim).to(device) for i in range(num_heads)]
        self.w_o=nn.Linear(embed_dim*num_heads,embed_dim).to(device)
        self.softmax=nn.Softmax().to(device)
    def single_head(self,q,k,v,head_idx):
        '''scale dot-scale attention '''
        q=self.w_q[head_idx](q)
        k=self.w_k[head_idx](k)
        v=self.w_v[head_idx](v)
        out=torch.matmul(torch.matmul(q,k.permute(0,2,1)),v)/self.embed_dim
        return out
    def forward(self,q,k,v):
        output=[]
        for i in range(self.num_heads):
            out=self.single_head(q,k,v,i)
            output.append(out)
        output=torch.cat(output,dim=2)
        output=self.w_o(output)
        # print(output.shape)
        return output

#ppt模型
class model_prompt(nn.Module):
    def __init__(self,batch_size,prompt_length,embedding_length,prompt_position,prompt_cls_length,prompt_cls_position,token_length,prompt_coop,attention_num):
        super(model_prompt,self).__init__()
        self.prompt_length=prompt_length
        self.token_length=token_length
        self.prompt_position = prompt_position
        self.embedding_length=embedding_length
        self.prompt_cls_length=prompt_cls_length
        self.prompt_cls_position=prompt_cls_position
        # self.prompt_coop_embedding = net.token_embedding(clip.tokenize([prompt_coop]).to(device)).squeeze(0)[
        #                         1:1 + prompt_length]
        # self.prompt = nn.Parameter(self.prompt_coop_embedding)
        self.prompt = nn.Parameter(torch.zeros((self.prompt_length, self.embedding_length)))
        torch.nn.init.normal_(self.prompt,mean=0,std=0.02)
        # self.prompt_cls = nn.Parameter(torch.zeros((self.prompt_cls_length, self.embedding_length)))
        self.norm = nn.BatchNorm1d(batch_size,affine=False)
        self.attn=attention(self.embedding_length,attention_num).to(device)
        self.linear3=nn.Linear(self.embedding_length,self.embedding_length).to(device)
        self.relu=nn.ReLU().to(device)
        self.linear4=nn.Linear(self.embedding_length,self.embedding_length).to(device)

    def forward(self,name_num,image_features,hard_prompt_front,hard_prompt_back,name_classes):

        image_features=net.encode_image(image_features)
        batch_size = image_features.shape[0]
        image_features1=image_features.expand(1,batch_size,self.embedding_length)

        name_token = [name_classes[i] for i in name_num]#得到对应文件名
        name_token = [clip.tokenize("{} {} {}".format(hard_prompt_front,name,hard_prompt_back)) for name in name_token]#tokenembedding
        name_token = torch.tensor([item.cpu().detach().numpy() for item in name_token])#转化为tensor
        name_token = torch.squeeze(name_token,dim=1)
        name_token = name_token.to(device)

        name_token1 = net.token_embedding(name_token)
        prompt=self.prompt.to(device)
        # prompt=prompt.expand(batch_size,self.prompt_length,self.embedding_length)

        #左侧注意力
        attns=self.attn(image_features1, image_features1, image_features1)
        attns = self.norm(attns)
        features = self.relu(self.linear3(attns))

        features=self.linear4(features)
        feature_front=features.expand(self.prompt_length,batch_size,self.embedding_length)
        feature_front=feature_front.permute(1,0,2)
        prompt=feature_front + prompt

        # prompt_cls = self.prompt_cls.to(device)
        # prompt_cls = prompt_cls.expand(1, self.prompt_cls_length, self.embedding_length)
        #
        # feature_back=features.expand(1,batch_size,self.embedding_length)
        # feature_back=feature_back.permute(1,0,2)
        # prompt_cls=prompt_cls+feature_back

        name_token1 = name_token1 +net.positional_embedding#batch_size*77*embedding_length
        name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]= prompt#+name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]
        # for i in range(batch_size):
        #     name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]= \
        #     prompt_cls+name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]

        x = name_token1.permute(1, 0, 2)
        out = net.transformer(x)
        out = out.permute(1, 0, 2)
        out1 = net.ln_final(out)
        text_features = out1[torch.arange(out1.shape[0]), name_token.argmax(dim=-1)] @ net.text_projection

        return text_features,image_features#,self.softmax(self.linear4(features))

class model_prompt_ba(nn.Module):
    def __init__(self,batch_size,prompt_length,embedding_length,prompt_position,prompt_cls_length,prompt_cls_position,token_length,prompt_coop,attention_num):
        super(model_prompt_ba,self).__init__()
        self.prompt_length=prompt_length
        self.token_length=token_length
        self.prompt_position = prompt_position
        self.embedding_length=embedding_length
        self.prompt_cls_length=prompt_cls_length
        self.prompt_cls_position=prompt_cls_position
        # self.prompt_coop_embedding = net.token_embedding(clip.tokenize([prompt_coop]).to(device)).squeeze(0)[
        #                         1:1 + prompt_length]
        # self.prompt = nn.Parameter(self.prompt_coop_embedding)
        self.prompt = nn.Parameter(torch.zeros((self.prompt_length, self.embedding_length)))
        torch.nn.init.normal_(self.prompt,mean=0,std=0.02)
        # self.prompt_cls = nn.Parameter(torch.zeros((self.prompt_cls_length, self.embedding_length)))
        self.norm = nn.BatchNorm1d(batch_size,affine=False)
        self.attn=attention(self.embedding_length,attention_num).to(device)
        self.linear3=nn.Linear(self.embedding_length,self.embedding_length).to(device)
        self.relu=nn.ReLU().to(device)
        self.linear4=nn.Linear(self.embedding_length,self.embedding_length).to(device)

    def forward(self,name_num,image_features,hard_prompt_front,hard_prompt_back,name_classes):

        image_features=net.encode_image(image_features)
        batch_size = image_features.shape[0]
        image_features1=image_features.expand(1,batch_size,self.embedding_length)

        name_token = [name_classes[i] for i in name_num]#得到对应文件名
        name_token = [clip.tokenize("{} {} {}".format(hard_prompt_front,name,hard_prompt_back)) for name in name_token]#tokenembedding
        name_token = torch.tensor([item.cpu().detach().numpy() for item in name_token])#转化为tensor
        name_token = torch.squeeze(name_token,dim=1)
        name_token = name_token.to(device)

        name_token1 = net.token_embedding(name_token)
        prompt=self.prompt.to(device)
        # prompt=prompt.expand(batch_size,self.prompt_length,self.embedding_length)

        #左侧注意力
        # attns=self.attn(image_features1, image_features1, image_features1)
        # attns = self.norm(attns)
        features = self.relu(self.linear3(image_features1))

        features=self.linear4(features)
        feature_front=features.expand(self.prompt_length,batch_size,self.embedding_length)
        feature_front=feature_front.permute(1,0,2)
        prompt=feature_front + prompt

        # prompt_cls = self.prompt_cls.to(device)
        # prompt_cls = prompt_cls.expand(1, self.prompt_cls_length, self.embedding_length)
        #
        # feature_back=features.expand(1,batch_size,self.embedding_length)
        # feature_back=feature_back.permute(1,0,2)
        # prompt_cls=prompt_cls+feature_back

        name_token1 = name_token1 +net.positional_embedding#batch_size*77*embedding_length
        name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]= prompt#+name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]
        # for i in range(batch_size):
        #     name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]= \
        #     prompt_cls+name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]

        x = name_token1.permute(1, 0, 2)
        out = net.transformer(x)
        out = out.permute(1, 0, 2)
        out1 = net.ln_final(out)
        text_features = out1[torch.arange(out1.shape[0]), name_token.argmax(dim=-1)] @ net.text_projection

        return text_features,image_features#,self.softmax(self.linear4(features))

class model_prompt_init_0(nn.Module):
    def __init__(self,batch_size,prompt_length,embedding_length,prompt_position,prompt_cls_length,prompt_cls_position,token_length,prompt_coop,attention_num):
        super(model_prompt_init_0,self).__init__()
        self.prompt_length=prompt_length
        self.token_length=token_length
        self.prompt_position = prompt_position
        self.embedding_length=embedding_length
        self.prompt_cls_length=prompt_cls_length
        self.prompt_cls_position=prompt_cls_position
        # self.prompt_coop_embedding = net.token_embedding(clip.tokenize([prompt_coop]).to(device)).squeeze(0)[
        #                         1:1 + prompt_length]
        # self.prompt = nn.Parameter(self.prompt_coop_embedding)
        self.prompt = nn.Parameter(torch.zeros((self.prompt_length, self.embedding_length)))
        # torch.nn.init.normal_(self.prompt,mean=0,std=0.02)
        # self.prompt_cls = nn.Parameter(torch.zeros((self.prompt_cls_length, self.embedding_length)))
        self.norm = nn.BatchNorm1d(batch_size,affine=False)
        self.attn=attention(self.embedding_length,attention_num).to(device)
        self.linear3=nn.Linear(self.embedding_length,self.embedding_length).to(device)
        self.relu=nn.ReLU().to(device)
        self.linear4=nn.Linear(self.embedding_length,self.embedding_length).to(device)

    def forward(self,name_num,image_features,hard_prompt_front,hard_prompt_back,name_classes):

        image_features=net.encode_image(image_features)
        batch_size = image_features.shape[0]
        image_features1=image_features.expand(1,batch_size,self.embedding_length)

        name_token = [name_classes[i] for i in name_num]#得到对应文件名

        name_token = [clip.tokenize("{} {} {}".format(hard_prompt_front,name,hard_prompt_back)) for name in name_token]#tokenembedding
        name_token = torch.tensor([item.cpu().detach().numpy() for item in name_token])#转化为tensor
        name_token = torch.squeeze(name_token,dim=1)
        name_token = name_token.to(device)

        name_token1 = net.token_embedding(name_token)
        prompt=self.prompt.to(device)
        # prompt=prompt.expand(batch_size,self.prompt_length,self.embedding_length)

        #左侧注意力
        attns=self.attn(image_features1, image_features1, image_features1)
        attns = self.norm(attns)
        features = self.relu(self.linear3(attns))
        features=self.linear4(features)
        feature_front=features.expand(self.prompt_length,batch_size,self.embedding_length)
        feature_front=feature_front.permute(1,0,2)
        prompt=feature_front + prompt

        # prompt_cls = self.prompt_cls.to(device)
        # prompt_cls = prompt_cls.expand(1, self.prompt_cls_length, self.embedding_length)
        #
        # feature_back=features.expand(1,batch_size,self.embedding_length)
        # feature_back=feature_back.permute(1,0,2)
        # prompt_cls=prompt_cls+feature_back

        name_token1 = name_token1 +net.positional_embedding#batch_size*77*embedding_length
        name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]= prompt#+name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]
        # for i in range(batch_size):
        #     name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]= \
        #     prompt_cls+name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]

        x = name_token1.permute(1, 0, 2)
        out = net.transformer(x)
        out = out.permute(1, 0, 2)
        out1 = net.ln_final(out)
        text_features = out1[torch.arange(out1.shape[0]), name_token.argmax(dim=-1)] @ net.text_projection

        return text_features,image_features#,self.softmax(self.linear4(features))

class model_prompt_init_p(nn.Module):
    def __init__(self,batch_size,prompt_length,embedding_length,prompt_position,prompt_cls_length,prompt_cls_position,token_length,prompt_coop,attention_num):
        super(model_prompt_init_p,self).__init__()
        self.prompt_length=prompt_length
        self.token_length=token_length
        self.prompt_position = prompt_position
        self.embedding_length=embedding_length
        self.prompt_cls_length=prompt_cls_length
        self.prompt_cls_position=prompt_cls_position
        # self.prompt_coop_embedding = net.token_embedding(clip.tokenize([prompt_coop]).to(device)).squeeze(0)[
        #                         1:1 + prompt_length]
        # self.prompt = nn.Parameter(self.prompt_coop_embedding)
        self.prompt = nn.Parameter(torch.zeros((self.prompt_length, self.embedding_length)))
        torch.nn.init.normal_(self.prompt,mean=0,std=0.02)
        # self.prompt_cls = nn.Parameter(torch.zeros((self.prompt_cls_length, self.embedding_length)))
        self.norm = nn.BatchNorm1d(batch_size,affine=False)
        self.attn=attention(self.embedding_length,attention_num).to(device)
        self.linear3=nn.Linear(self.embedding_length,self.embedding_length).to(device)
        self.relu=nn.ReLU().to(device)
        self.linear4=nn.Linear(self.embedding_length,self.embedding_length).to(device)

    def forward(self,name_num,image_features,hard_prompt_front,hard_prompt_back,name_classes):

        image_features=net.encode_image(image_features)
        batch_size = image_features.shape[0]
        image_features1=image_features.expand(1,batch_size,self.embedding_length)

        name_token = [name_classes[i] for i in name_num]#得到对应文件名
        name_token = [clip.tokenize("{} {} {}".format(hard_prompt_front,name,hard_prompt_back)) for name in name_token]#tokenembedding
        name_token = torch.tensor([item.cpu().detach().numpy() for item in name_token])#转化为tensor
        name_token = torch.squeeze(name_token,dim=1)
        name_token = name_token.to(device)

        name_token1 = net.token_embedding(name_token)
        prompt=self.prompt.to(device)
        # prompt=prompt.expand(batch_size,self.prompt_length,self.embedding_length)

        #左侧注意力
        attns=self.attn(image_features1, image_features1, image_features1)
        attns = self.norm(attns)
        features = self.relu(self.linear3(attns))
        features=self.linear4(features)
        feature_front=features.expand(self.prompt_length,batch_size,self.embedding_length)
        feature_front=feature_front.permute(1,0,2)
        prompt=feature_front + prompt

        # prompt_cls = self.prompt_cls.to(device)
        # prompt_cls = prompt_cls.expand(1, self.prompt_cls_length, self.embedding_length)
        #
        # feature_back=features.expand(1,batch_size,self.embedding_length)
        # feature_back=feature_back.permute(1,0,2)
        # prompt_cls=prompt_cls+feature_back

        name_token1 = name_token1 +net.positional_embedding#batch_size*77*embedding_length
        name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]= prompt#+name_token1[:,self.prompt_position:self.prompt_length+self.prompt_position,:]
        # for i in range(batch_size):
        #     name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]= \
        #     prompt_cls+name_token1[i,self.prompt_cls_position+classes[name_num[i]]: self.prompt_cls_position + self.prompt_cls_length+classes[name_num[i]],:]

        x = name_token1.permute(1, 0, 2)
        out = net.transformer(x)
        out = out.permute(1, 0, 2)
        out1 = net.ln_final(out)
        text_features = out1[torch.arange(out1.shape[0]), name_token.argmax(dim=-1)] @ net.text_projection

        return text_features,image_features#,self.softmax(self.linear4(features))

#训练函数
def train(model,train_loader,hard_prompt_front,hard_prompt_back,epoch,name_classes,optimizer,logger):

    Loss = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    for j in range(epoch):
        for i,test_data in enumerate(train_loader):

            name_token=test_data[1]
            image_features=test_data[0]
            batch_size=image_features.shape[0]
            name_token=name_token.to(device)
            image_features=image_features.to(device)
            labels = torch.eye(batch_size).to(device)
            logit_scale = net.logit_scale.exp()

            with autocast():
                text_features, image_features = model(name_token, image_features, hard_prompt_front, hard_prompt_back,
                                                      name_classes)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()

                loss = Loss(logits, labels)  # +loss2
            # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # logger.info('epoch=' +str(j)+'  loss=' +str(loss.item()))

def test(model,test_loader,classes_name,class_min,hard_prompt_front,hard_prompt_back,class_num):
    right_number=0
    number = 0
    with torch.no_grad():
        for j,test_data in enumerate(test_loader):
            images_test=test_data[0].to(device)

            image_features = net.encode_image(images_test)#图片向量batch_size * embedding_length
            test_batch_size=image_features.shape[0]
            # print(hard_prompt_front,hard_prompt_back)

            name_token = [clip.tokenize("{} {} {}".format(hard_prompt_front,name,hard_prompt_back)) for name in classes_name]  # tokenembedding
            # print(["{} {} {}".format(hard_prompt_front, name, hard_prompt_back) for name in classes_name])
            name_token = torch.tensor([item.cpu().detach().numpy() for item in name_token])  # 转化为tensor
            name_token = torch.squeeze(name_token, dim=1).to(device)
            name_token1 = net.token_embedding(name_token)#class_num * token_length * token_embedding
            #左侧
            prompt = model.prompt.to(device)
            prompt = prompt.expand(test_batch_size,model.prompt_length, model.embedding_length)#class_num * prompt_length * embedding_length

            #左侧attention模块
            image_features1 = image_features.expand(1, test_batch_size, model.embedding_length)
            attns = model.attn(image_features1, image_features1, image_features1)#注意力层
            features = model.linear4(model.relu(model.linear3(attns)))#线性层
            features=features.permute(1,0,2)
            prompt = prompt + features#将图片向量加入提示中
            prompt = prompt.expand(class_num, test_batch_size, model.prompt_length, model.embedding_length)
            name_token1 = name_token1 + net.positional_embedding#class_num*token_length * token_embedding
            name_token1=name_token1.expand(test_batch_size,class_num,model.token_length, model.embedding_length)
            name_token1=name_token1.permute(1,0,2,3)
            name_token_true=name_token1.clone()

            name_token_true[:, :,model.prompt_position:model.prompt_length + model.prompt_position, :] = prompt
            x = name_token_true.permute(1,2, 0, 3)
            text_features=[]
            for num,z in enumerate(x):
                out = net.transformer(z)
                out = out.permute(1, 0, 2)
                out1 = net.ln_final(out)
                text_features.append(out1[torch.arange(out1.shape[0]), name_token.argmax(dim=-1)] @ net.text_projection)
            get_class=get_classes(image_features,text_features,class_min)

            labels_test = test_data[1]
            get_class = torch.tensor(get_class)
            number += len(labels_test)
            right_number+=sum(labels_test==get_class)
    return right_number/number

def get_classes(image_features,text_features,class_min_):
    get_class=[]
    for index,image_feature in enumerate(image_features):
        simi=[]
        for text_feature in text_features[index]:
            simi.append(cosine_similarity(image_feature.detach().cpu().numpy(),text_feature.detach().cpu().numpy()))
        get_class.append(np.argmax(simi)+class_min_)
    return get_class

def cosine_similarity(a,b):
    a=a.reshape(-1)
    b=b.reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


