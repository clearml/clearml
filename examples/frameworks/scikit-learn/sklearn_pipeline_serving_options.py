# ClearML - Example of registering a scikit-learn pipeline and storing its serving options
#
# The pipeline holds the input preparation (scaling) and the classifier as one object, so the saved model
# takes the raw measurements. The model is registered with OutputModel.update_weights, and the options
# ClearML Serving needs to serve it are stored with the model.
import joblib
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clearml import OutputModel, Task


# The model is registered below with update_weights, so automatic registration of joblib saves is
# turned off: the model is registered once. output_uri=True uploads the model file
task = Task.init(
    project_name="examples",
    task_name="Scikit-learn pipeline serving options",
    output_uri=True,
    auto_connect_frameworks={"joblib": False},
)

# The wine measurements have very different ranges, so the classifier needs them scaled
wine = datasets.load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

# The scaler and the classifier are fitted together, saved together and used together
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("Test accuracy: {:.2f}".format(model.score(X_test, y_test)))

joblib.dump(model, "wine_pipeline.pkl")

# Register the file, then store how the model should be served: the options of
# `clearml-serving model add`, so they do not have to be given again at deployment
output_model = OutputModel(task=task, name="wine pipeline", framework="ScikitLearn")
output_model.update_weights(weights_filename="wine_pipeline.pkl")
output_model.set_serving_engine("sklearn")
output_model.set_serving_endpoint("wine")
output_model.set_serving_input_example(X_test[:5])
# the packages the model needs when it is served: a pipeline is loaded by the scikit-learn that saved it
output_model.set_serving_requirements(["scikit-learn=={}".format(sklearn.__version__)])
