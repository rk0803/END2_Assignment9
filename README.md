# END2_Assignment9
Assignment 9 is about implementing different metrics:
1. Precision, Recall and F1 Score
2. Bleu Score
3. Perplexity
4. BERT Score

Notebook **Assignment9_1PRF1.ipynb** contains the implementation of Metric 1, i.e. Precision, Recall and F1 Score </br>
Notebook **Assignment9_2BLEU.ipynb** uses the implementation of BLEU Score </br>
Notebook **Assignment9_3pplquora.ipynb** contains the implementation of Metric 3 i.e.  Perplexity </br>
Metric  4 is not implemented. </br>
Here is discussion on these. </br>
## Metrics
### Precision, Recall and F1 Score 
_Note_: For a more mathematical representation, please look at the documentation for same in the notebook **Assignment9_PRF1.ipynb**
Here I have taken a multiclass classification problem (Stanford Sentiment Aanalysis).
There are Five classes (0,1,2,3,4). </br>
The count is these classes in the dataset, gives the actual number of examples for each class.</br>
Then we make predictions using our model and categorize each sample into one of the five different classes. The count of these classes in the predictions gives the predicted count for each class.</br>
The confusion matrix in this case will look something like this</br>
![image](https://user-images.githubusercontent.com/82941475/124909312-ecf7fc80-e007-11eb-9e35-03a667844277.png)


Here **tj** represents actual class is **j** and predicted class is also **j** </br>
and **fij** represents actual class is **i** and predicted (falsely) class is **j**  </br>
 **Precision** is defined as Number of correct predictions of a class out of all the predictions of that class. </br>
 So for  **j**th class, total false predicted, false positives **fpj** by summing over all **fij**, for **i** not equal **j**.</br> 
 ![image](https://user-images.githubusercontent.com/82941475/124909627-4d873980-e008-11eb-9610-8c2e65048640.png)
 Precision = (tj)/(t + fpj)  </br>
 
 **Recall** is defined as Number of correct predictions of a class out of actual instances in the dataset for that class.</br>
So for **j**th class, total   instances for  **j** th class, which are incorrectly predicted,i.e. **fnj** is calculated by summing over all **fji**, for **i** not equal **j** </br>
![image](https://user-images.githubusercontent.com/82941475/124909708-67288100-e008-11eb-884b-917d9f1d8f10.png)
 Recall =(tj)/(tj + fnj) </br>
 **F1 Score** is the weigthed average of precision and recall and is defined as the harmonic mean of the two. so </br>
 F1 score= (2 * precision * recall )/(precision + recall)  </br>
 </br>
Since its a multiclass classification, _precision, recall_ and _F1 score_ are defined for each class.
In terms of semantics, precision tells us, out of the samples predicted for a class, how many actually belonged to that class. Similarly, recall tells us, how sensitive our system is towards correctly predicted the class. 
To be able to compare different models based on both the scores, we take harmonic mean of the two (weighted average) and that is what F1 score tells us. </br>
The question now arises, which one to use and when. </br></br>
Based on the meaning of precision and meaning of the class labels, if we want false predictions for that label as low as possible we should use precision.
E.g if class label 0 means ver negative, and we dont want to miss any of the very negative reviews (so as to be able to act on negative remarks in the review), we should have **fn0** as low as possible, i.e. recall as high as possible.
Now, what would be situations where we want **fpj** to be as low as possible? So in our scenario, if most of the reviews are predicted as 0, which is very negative, where as they are actually not class 0, it may tarnish the image of the restaurant/ movies and lead to revenue loss, we should worry about keeping **fpj** as low as possible, i.e. precision as high as possible. 

Now in our scenario, all the classes are equally important, i.e. there is not a cost associated for misclassification of a certain class. So, I have taken the average of the precision and recall, and that becomes the precision and recall of the model.
Using these values of precision and recall, I have calculated the F1 Score.

### BLEU Score
Bilgual Evaluation Understudy (BLEU) is a metric for automatically evaluating machine-translated text. The BLEU score is a number between zero and one that measures the similarity of the machine-translated text to a set of high quality reference translations. A value of 0 means that the machine-translated output has no overlap with the reference translation (low quality) while a value of 1 means there is perfect overlap with the reference translations (high quality). The general guideline for interpreting the BLEU score, as published on google cloud is as below: </br>

##### BLEU Score	(in percent) Interpretation
Score | Interpretation
-------|-------------
< 10	| Almost useless
10 - 19	| Hard to get the gist
20 - 29	| The gist is clear, but has significant grammatical errors
30 - 40	| Understandable to good translations
40 - 50	 | High quality translations
50 - 60	| Very high quality, adequate, and fluent translations
> 60	| Quality often better than human

BLEU is a Corpus-based Metric and doesn't generall perform well to evaluate single sentences. It also does not distinguish between content and function words, that is, a dropped function word like "a" gets the same penalty as if the name "NASA" were erroneously replaced with "ESA". It should not be used when evaluating the translation on semantic structures. Prior to computing the BLEU score, both the reference and candidate translations are normalized and tokenized. The choice of normalization and tokenization steps can  affect the final BLEU score to a large extent.

### Perplexity
*Note*  For a more mathematical representation, please look at the documentation for same in the notebook **Assignment9_3pplquora.ipynb**

In the context of Natural Language Processing, perplexity is one way to evaluate language models. is not just enough to produce text; we also need a way to measure the quality of the produced text. One such way is to measure how surprised or **perplexed** the RNN was to see the output given the input. That is, if the cross-entropy loss for an input  **xi**  and its corresponding output  **yi**  is  **loss(xi,yi)**  , then the perplexity would be as follows: ![formula](https://render.githubusercontent.com/render/math?math=\P(xi,yi)=e^{loss(xi,yi)})
Using this, we can compute the average perplexity for a training dataset of size M as:  ![formula](https://render.githubusercontent.com/render/math?math=\PPL(Datasettrain)=\frac{1}{M}\sum_{i}^{M}P(xi,yi))

Question now is what is cross entropy?
Lets first understand the definition of entropy given by Shannon.
Shannon's Entropy is defined as  ![formula](https://render.githubusercontent.com/render/math?math=\E(p)=-\sum_{i}^{n}x_{i}log_{b}p(x_{i}))
where  b  is the base of logarithm used,  n  is the number of states, and  p(xi)  is the probability of system being in state  i , and 
![formula](https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{n}p(x_{i})=1).

So, Shannon entropy tells us that the if a system can be in, say, four possible states, and we know the probability of the system being in any one of the states, then for an infintely long sequence of states, how much minimum memory do we need to store the state of the system.

Now Lets look at cross entropy.
As the word 'cross' implies, we have two different distributions, say  p  and  q , for the system to be in those possible states, then cross entropy  
![formula](https://render.githubusercontent.com/render/math?math=\CE(p,q)=-\sum_{i}^{n}x_{i}log_{b}q(x_{i}))
So, lets say we have two different systems say  S1  and  S2 , with two different probability distributions  p  and  q . Then cross entropy tells us, for an infinitely large sequence of states, drawn from system  S1  with probability distribution  p  and from system  S2 , with probability distribution  q , how much minimum memory do we need on average to store the states.
E(p)  will always be less than cross entropy. If  p=q ,  CE(p,q)  will be equal to  E(p) , and will be at its minimum value.

Now lets look at Perplexity.
Preplexity is defined with cross entropy as :
![formula](https://render.githubusercontent.com/render/math?math=\PPL(p)=b^{CE(p,q)})
But what is the pupose of Perplexity in language modeling?
If we take  M  different sentences in the dataset, then these  M  different sequences represent  m  different possible states (Some of them same). Now we are building the language model, the original system has states distributed with probability  p , which have no way to know. We can only estimate that probability distribution by, say,  q . Then the cross_entropy is  CE(p,q) , which we also call as cross entropy loss. So for each sequence of states (i.e. sentences), we can write that as  loss(xi,yi)=CE(p(xi),q(yi)) , where  xi  comes from the original system  S1  and  yi  comes from the system  S2 , which is the system we are trying to model for  S1 .

If the language model (the one we are bulding) is of extremely low quality, and the words in the sentence are guessed randomly, with each word chosen in equally likely manner, then   ![formula](https://render.githubusercontent.com/render/math?math=\q(w_{i}|w_{1},w_{2}...w_{i-1})=\frac{1}{m}) , log of this number will be very high ( m  being very large, making  ![formula](https://render.githubusercontent.com/render/math?math=\\frac{1}{m})  very small), and hence CE will be very high, leading to high perplexity.

But if a model is better, and has actually learned something, then the probability  q  of a valid sentence like "I like apples" is very high (hence log of that very small, hence small preplexity), as compared to an invalid sentence like " apple fly state".

So as the model learns with each epoch, making its probability distribution  q , closer to the actual distribution  p , loss reduces and so is the preplexity. So lower the perplxity better is the model.

### BERT Score
BERTScore computes a similarity score for each token in the candidate sentence with each token in the reference sentence. However, instead of exact matches, we compute token similarity using contextual embeddings. We evaluate using the outputs of 363 machine translation and image captioning systems. BERTScore correlates better with human judgments and provides stronger model selection performance than existing metrics
According to a research study, there are two major drawbacks in n-gram-based metrics. to be able to judge effectiveness of praphrasing. For example, given the reference, *"people like foreign cars"*, BLEU score often gives a higher score to *"people like visiting places abroad"* instead of  *"consumers prefer imported cars"*. Here, even though the sentence is semantically correct, it gets a lower score and performance of the model is underestimated. Thus, instead of counting tokens matching, as in BLEU, we compute similarity score is calculated using contextualized token embeddings, which works better if there needs to be paraphrase detection, in general. Apart from this, n-gram model evaluation techniques, cannot take into account dependencies which are more than n (n as in n-gram) distance apart, and hence give a lower score to sentences which are semanticalyy re-structured. For example, if we are taking a 2-gram model, , BLEU will only mildly penalize swapping of cause and effect clauses (e.g. A because B instead of B because A), especially when the arguments A and B are long phrases. This is especially ineffective in domains like legal domains.  </br>
Bert score, since it calculates the similarity score based on contextual embedding assigns a better performance to the system. </br>
The similarity matrix used here is cosine similarity between the generated sentence and actual sentence, (treating them as vectors of dimension k, k being the length of the sentence. This is done by taking the dot product of the two. By definition if we have A and B as two vectors, then 
![image](https://user-images.githubusercontent.com/82941475/125023307-37c35400-e09c-11eb-94e6-50c633a8a35d.png). </br>
 If the angle between two vectors is very small (i.e. if they are similar), ![formula](https://render.githubusercontent.com/render/math?math=\cos(\theta)) will be close to one. Here is the complete radial for the different values of  ![formula](https://render.githubusercontent.com/render/math?math=\cos(\theta)) and what it means.</br>
 ![image](https://user-images.githubusercontent.com/82941475/125023431-896bde80-e09c-11eb-9dba-65f7a4c4b742.png) </br>

## Implementation and Discussion
### Precision, Recall and F1 Score 
```code
#Function to calculate Precision, Recall and F1 score.

def cal_prf1(pred,y):
    tc=torch.zeros(len(Label.vocab)).to(device) #tc gives us how many are true positive for a class
    fc=torch.zeros(len(Label.vocab)).to(device) #fc gives us how many are false positive for a class
    top_pred=pred.argmax(1,keepdim=True)
    top_pred=top_pred.squeeze()
    for i,j in enumerate(top_pred):
      if j==y[i]:
        tc[j]+=1
      else:
        fc[j]+=1
    #print(tc, fc)
    #To avoid 0/0 kind of scenario, 0 is replaced with a very small number 0.00001
    rep=torch.tensor(0.00001).to(device) 
    tc=torch.where(tc==0,rep,tc)
    fc=torch.where(fc==0,rep,fc)
    #pc gives class-wise precision 
    pc=tc/(tc+fc) 
    y=y.int()
    #y.bincount() gives the total number of instances in every class as y is the actual labels
    #print(y.bincount())
    #rc gives class-wise recall 
    rc=tc/y.bincount()
    # The average of precision and recall gives the precision and recall for the model. 
    #Here I have given equal weights to each class. But it can be a weighted average.
    p=pc.sum()/len(pc)
    r=rc.sum()/len(rc)
    f1=2*p*r/(r+p)
    return p,r,f1
```
#### Training log 
Here is the training log for first five epochs: </br>
epoch | Train Loss | Train Acc | Train Precision | Train Recall | Train F1 Score | Test Loss | Test Acc | Test Precision | Test Recall | Test F1 Score
-----|----------|---------|----------------|-------------|--------------|--------------|--------------|----------------|------|--------
  1 |	 4892.785 |  25.66% | 0.242 | 0.21 | 0.2223016 |	4692.892 | 25.39% |	 0.242 | 0.21 |     0.2223016 
  2 |	4506.224 |  27.02% |	  0.360 |0.23 | 0.2804562 |	4323.563 |   28.18% |	 0.360 | 0.23 |   0.2804562 
  3 |	4152.384 | 29.80%|	 0.371 | 0.22 |   0.2741793 |	 3984.663 |  25.85% |	 0.371 | 0.22 |   0.2741793 
  4 | 3827.110 | 31.74% |	 0.392 | 0.29 |  0.3329042 |	3672.595 |  34.24%  |	  0.392 |  0.29 | 0.3329042 
  5 |	 3527.245 | 35.30% |	0.407 | 0.29 |  0.3399828 |	 3384.608 |  33.83%  |	0.407 | 0.29 |   0.3399828 
  
 ####  Results discussion
 As can be seen as the model learns aligns itself to the training data ( i.e loss reduces), precision, recall and hence f1 score become better for the training data. When the model is evaluated on the test data (unseen data) , the results are comparable.
### BLEU Score
Bleu Score directly is not implemented but is used, while translating from German to English. Here is a snapshot of the training log.
#### Training Log

epoch | Train Loss | Train PPL | Valid Loss | Valid PPL | Train Bleu Score | Test Bleu Score 
-----|----------|---------|----------------|-----------|--------|--------
1    |  3.253 |  25.875 | 3.900 |   49.421 |   0.169 |    0.152
2 | 2.779 |  16.108 | 3.684 |   39.814 |	 0.198 |  0.172

It was a less than optimal implementation and each epoch was taking around 20 minutes. After 2 epochs error started coming. I will have to investigate further on this.
But it can be seen that in second epoch % bleu score moved from 16.9 to 19.8.  This is just inside the range where the model can be said to be not able to get the gist.

### Perplexity

```code
#Code for perplexity
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
def prplxity(l):
  ppl=math.exp(l)
  return ppl
```
#### Training Log
epoch | Train Loss | Train PPL |  Test Loss | Test PPL
-----|----------|---------|----------------|-------------
 01 |  2.990 |  19.893 | 	3.761 |   42.991
 02 |  2.753 |  15.688  |	3.672 |  39.312
 03 |  2.576 |  13.147 |	  3.580 |   35.856
 04 | 2.433 |   11.396 | 	  3.599 |    36.571
 05 |  2.303 |  10.003 | 	  3.581 |    35.924
#### Discussion
The loss curve plot below shows that training loss steadliy goes down while the test loss fluctuates. </br>
![image](https://user-images.githubusercontent.com/82941475/125026619-9f7c9d80-e0a2-11eb-9013-9599931e5c45.png) </br>
The Perplexity  curve plot below shows that training perplexity steadliy goes down while the test perplexity fluctuates, in line with loss results.</br>
![image](https://user-images.githubusercontent.com/82941475/125026762-de125800-e0a2-11eb-98bb-5b70932012df.png) </br>

It can be seen that the train perplexity is gradully reducing with each epoch, as the model is getting better in predicting the text based on what it has seen. So it slowly aligns itself with seen examples and adjusts the probabilites better. So the cross entropy loss reducing leading to a low value for training perplexity. However, the validation perplexity is fluctuating significantly. This is expected because what we are essentially evaluating in the validation perplexity is our model's ability to predict a unseen text based on our learning on training data. Since language can be quite difficult to model, this is a very difficult task, and these fluctuations are natural.

Also it can be seen that fluctuations in perplexity are in line with fluctuations in the loss curve.

### BERT Score
I have not implemented this metric.

#### Team Members
Ritambhra Korpal </br>
Chaitanya Vanapamala </br>
Pralay Ramteke </br>
Pallavi</br>
