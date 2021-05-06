# @hidden_cell

from ibm_watson_machine_learning.helpers import DataConnection
from ibm_watson_machine_learning.helpers import S3Connection, S3Location

training_data_reference = [DataConnection(
    connection=S3Connection(
        api_key='Pk1MWfia1imWVbbBmCeCDv8zmg39aR2p96nqb0zIJaEq',
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net'
    ),
    location=S3Location(
        bucket='heartdiseaseprediction-donotdelete-pr-txewlfjge6vtwz',
        path='Heart_Disease_Prediction.csv'
    )),
]
training_result_reference = DataConnection(
    connection=S3Connection(
        api_key='Pk1MWfia1imWVbbBmCeCDv8zmg39aR2p96nqb0zIJaEq',
        auth_endpoint='https://iam.bluemix.net/oidc/token/',
        endpoint_url='https://s3.eu-geo.objectstorage.softlayer.net'
    ),
    location=S3Location(
        bucket='heartdiseaseprediction-donotdelete-pr-txewlfjge6vtwz',
        path='auto_ml/6c66620c-fbf3-496e-a48d-72a4bae0216f/wml_data/9364352c-02c7-4635-8e40-9c149991e8c0/data/automl',
        model_location='auto_ml/6c66620c-fbf3-496e-a48d-72a4bae0216f/wml_data/9364352c-02c7-4635-8e40-9c149991e8c0/data/automl/cognito_output/Pipeline1/model.pickle',
        training_status='auto_ml/6c66620c-fbf3-496e-a48d-72a4bae0216f/wml_data/9364352c-02c7-4635-8e40-9c149991e8c0/training-status.json'
    ))

experiment_metadata = dict(
    prediction_type='classification',
    prediction_column='Heart Disease',
    holdout_size=0.1,
    scoring='accuracy',
    deployment_url='https://eu-gb.ml.cloud.ibm.com',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_reference=training_data_reference,
    training_result_reference=training_result_reference,
    project_id='c6db1e2d-c3a5-4362-b5be-ad97ef7edbd5',
    positive_label='Absence',
    drop_duplicates=True
)

df = training_data_reference[0].read(csv_separator=experiment_metadata['csv_separator'])
df.dropna('rows', how='any', subset=[experiment_metadata['prediction_column']], inplace=True)

from sklearn.model_selection import train_test_split

df.drop_duplicates(inplace=True)
X = df.drop([experiment_metadata['prediction_column']], axis=1).values
y = df[experiment_metadata['prediction_column']].values

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=experiment_metadata['holdout_size'],
                                                    stratify=y, random_state=experiment_metadata['random_state'])

from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
from autoai_libs.transformers.exportable import boolean2float
from autoai_libs.transformers.exportable import CatImputer
from autoai_libs.transformers.exportable import CatEncoder
import numpy as np
from autoai_libs.transformers.exportable import float32_transform
from sklearn.pipeline import make_pipeline
from autoai_libs.transformers.exportable import FloatStr2Float
from autoai_libs.transformers.exportable import NumImputer
from autoai_libs.transformers.exportable import OptStandardScaler
from sklearn.pipeline import make_union
from autoai_libs.transformers.exportable import NumpyPermuteArray
from autoai_libs.cognito.transforms.transform_utils import TA1
import autoai_libs.utils.fc_methods
from autoai_libs.cognito.transforms.transform_utils import FS1
from sklearn.linear_model import LogisticRegression

numpy_column_selector_0 = NumpyColumnSelector(
    columns=[1, 2, 5, 6, 8, 10, 11, 12]
)
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=[
        "float_int_num", "float_int_num", "float_int_num", "float_int_num",
        "float_int_num", "float_int_num", "float_int_num", "float_int_num",
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
    misslist_list=[[], [], [], [], [], [], [], []],
)
numpy_replace_missing_values_0 = NumpyReplaceMissingValues(
    missing_values=[], filling_values=100001
)
numpy_replace_unknown_values = NumpyReplaceUnknownValues(
    filling_values=100001,
    filling_values_list=[
        100001, 100001, 100001, 100001, 100001, 100001, 100001, 100001,
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
)
cat_imputer = CatImputer(
    strategy="most_frequent",
    missing_values=100001,
    sklearn_version_family="23",
)
cat_encoder = CatEncoder(
    encoding="ordinal",
    categories="auto",
    dtype=np.float64,
    handle_unknown="error",
    sklearn_version_family="23",
)
pipeline_0 = make_pipeline(
    numpy_column_selector_0,
    compress_strings,
    numpy_replace_missing_values_0,
    numpy_replace_unknown_values,
    boolean2float(),
    cat_imputer,
    cat_encoder,
    float32_transform(),
)
numpy_column_selector_1 = NumpyColumnSelector(columns=[0, 3, 4, 7, 9])
float_str2_float = FloatStr2Float(
    dtypes_list=[
        "float_int_num", "float_int_num", "float_int_num", "float_int_num",
        "float_num",
    ],
    missing_values_reference_list=[],
)
numpy_replace_missing_values_1 = NumpyReplaceMissingValues(
    missing_values=[], filling_values=float("nan")
)
num_imputer = NumImputer(strategy="median", missing_values=float("nan"))
opt_standard_scaler = OptStandardScaler(
    num_scaler_copy=None,
    num_scaler_with_mean=None,
    num_scaler_with_std=None,
    use_scaler_flag=False,
)
pipeline_1 = make_pipeline(
    numpy_column_selector_1,
    float_str2_float,
    numpy_replace_missing_values_1,
    num_imputer,
    opt_standard_scaler,
    float32_transform(),
)
union = make_union(pipeline_0, pipeline_1)
numpy_permute_array = NumpyPermuteArray(
    axis=0, permutation_indices=[1, 2, 5, 6, 8, 10, 11, 12, 0, 3, 4, 7, 9]
)
ta1_0 = TA1(
    fun=np.rint,
    name="round",
    datatypes=["numeric"],
    feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical],
    col_names=[
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
        "EKG results", "Max HR", "Exercise angina", "ST depression",
        "Slope of ST", "Number of vessels fluro", "Thallium",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"),
    ],
)
fs1_0 = FS1(
    cols_ids_must_keep=range(0, 13),
    additional_col_count_to_keep=12,
    ptype="classification",
)
ta1_1 = TA1(
    fun=np.sqrt,
    name="sqrt",
    datatypes=["numeric"],
    feat_constraints=[
        autoai_libs.utils.fc_methods.is_non_negative,
        autoai_libs.utils.fc_methods.is_not_categorical,
    ],
    col_names=[
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
        "EKG results", "Max HR", "Exercise angina", "ST depression",
        "Slope of ST", "Number of vessels fluro", "Thallium", "round(Age)",
        "round(BP)", "round(Cholesterol)", "round(Max HR)",
        "round(ST depression)",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
fs1_1 = FS1(
    cols_ids_must_keep=range(0, 13),
    additional_col_count_to_keep=12,
    ptype="classification",
)
logistic_regression = LogisticRegression(
    class_weight="balanced",
    multi_class="ovr",
    n_jobs=1,
    random_state=33,
    solver="liblinear",
)

pipeline = make_pipeline(
    union,
    numpy_permute_array,
    ta1_0,
    fs1_0,
    ta1_1,
    fs1_1,
    logistic_regression,
)
from sklearn.metrics import get_scorer

scorer = get_scorer(experiment_metadata['scoring'])

pipeline.fit(train_X, train_y)

score = scorer(pipeline, test_X, test_y)
