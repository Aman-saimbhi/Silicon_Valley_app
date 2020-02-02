# Silicon_Valley_app
Web app inspired from the hotdog app in the ''Silicon Valley'' show. Model is a fine tuned version of VGG-16 model which is deployed as a web service using flask.
Here is the link to the dataset https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog
As you can see that the dataset is quite small to train the complete deep CNN like VGG-16. So, I decided to make use of transfer learning to fine tune the model for my classification task. Loaded the paramaters which were trained on original dataset of 
very large size. Modified the last layer according to the needs of my classification task which was binary classification by freezing all the previous layers so that their parameters don't get trained.
Due to the unavailability of large dataset, the model was slightly overfitting to the training set. However, after some tweaking 
I was able to achieve 88% accuracy on the test set.

Model is deployed as a web service by making a REST api using flask. The backend of the web app can be found in predict_app.py 
python file. 
Front end web page is built to interact with the web service. An image is uploaded to the web page which is then converted into
base64image which is then made into json object which is then sent to the back end for futher processing.
Image is decoded and pre processed so that the predictions can be made on the image. From there the model predicts the result and
send it back to the front end as a json response. So this how the workflow of information is carried out.

You can clone this repository to make use of the service as well as the front end. Also it can be used to deploy any model
of your choice by making few necessary changes.

