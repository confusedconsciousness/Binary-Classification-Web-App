# Mushroom Classifier Web App
This repository contains code to deploy a web app on your local host using streamlit
We are using Mushrooms dataset for our classification purposes.<br>
<br> To view the web app please visit this link : https://mushroom-classifier-app.herokuapp.com/ </br>
In this web app we've used three classifiers
<ol>
  <li> Support Vector Machine</li>
  <li> Logistic Regression </li>
  <li> Random Forests </li>
</ol>
I've provided options in the web app to tune the hyperparameters itself. for e.g.<br>
<b> Support Vector Machine</b>
<ul>
  <li> C</li>
  <li> gamma </li>
  <li> Kernel </li>
</ul>
<b> Logistic Regression</b>
<ul>
  <li> C</li>
  <li> max_iter </li>
</ul>
<b> Random Forest</b>
<ul>
  <li> n_estimators</li>
  <li> max_depth</li>
  <li> bootstrap</li>
</ul>

This web app plots, three metrics to show the performance of these classifiers<br>
<ol>
  <li> Confusion Matrix </li>
  <li> ROC Curve</li>
  <li> Precision Recall Curve </li>
</ol>
To run the we app, open the terminal and navigate to this repository and type the following command<br>
```
streamlit run app.py
```
<br>This will open the we app in the browser.
  
