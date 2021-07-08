# END2_Assignment9
Assignment 9 is about implementing different metrics:
1. Precision, Recall and F1 Score
2. Bleu Score
3. Perplexity
4. BERT Score
Notebook **Assignment9_1PRF1.ipynb** contains the implementation of Metric 1, i.e. Precision, Recall and F1 Score </br>
Notebook **Assignment9_3pplquora.ipynb** contains the implementation of Metric 3 i.e.  </br>
Metric 2 and 4 are not implemented. </br>
Here is discussion on these. </br>
### Precision, Recall and F1 Score 
_Note_: For a more mathematical representation, please look at the documentation for same in the notebook **Assignment9_PRF1.ipynb**
Here I have taken a multiclass classification problem (Stanford Sentiment Aanalysis).
There are Five classes (0,1,2,3,4). </br>
The count is these classes in the dataset, gives the actual number of examples for each class.</br>
Then we make predictions using our model and categorize each sample into one of the five different classes. The count of these classes in the predictions gives the predicted count for each class.</br>
The confusion matrix in this case will look something like this</br>

   __ | Actual   
-----------------|------------------
**Predicted**    | 0,    1,    2,    3,    4
0| t0, f10, f20 ,f30, f40 
1| f01, t1, f21, f31, f41
2| f02, f12, t2, f32, f42
3 |f03, f13, f23, t3, f43
4| f04, f14, f24, f34, t4


Here **tj** represents actual class is **j** and predicted class is also **j** </br>
and **fij** represents actual class is **i** and predicted (falsely) class is **j**  </br>
 **Precision** is defined as Number of correct predictions of a class out of all the predictions of that class. </br>
 So for  **j**th class, total false predicted, false positives **fpj** by summing over all **fij**, for **i** not equal **j**.</br> 
 
 Precision = (tj)/(t + fpj)  </br>
 
 **Recall** is defined as Number of correct predictions of a class out of actual instances in the dataset for that class.</br>
So for **j**th class, total   instances for  **j** th class, which are incorrectly predicted,i.e. **fnj** is calculated by summing over all **fji**, for **i** not equal **j**.
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
