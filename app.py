import streamlit as st
import vtk
import numpy as np
import os
import matplotlib as plt
from PIL import Image
from vtk.util.numpy_support import vtk_to_numpy
from Run_model import load_model_saved,get_predicted_imgs
from dataframe_creator import get_img_data


st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache_resource

def load_model():
    model = load_model_saved()
    return model

root_dir = r"D:\Personal_User\BITS\third year\Design Project\TASK1\Dataset\train"
model = load_model()

st.markdown("<h1 style='text-align: center; color: White;'>Brain Tumor Segmentor</h1>",unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: White;'>All you have to do is Upload the MRI scan and the model will do the rest!</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: White;'>Submission for BraTS_MobileUNet</h4>", unsafe_allow_html=True)
st.sidebar.header(":green[What is this Project about?]")
st.sidebar.text("It is a Deep learning solution to detection of Brain Tumor using MRI Scans.")
st.sidebar.header(":green[What does it do?]")
st.sidebar.text("The user can upload their MRI scan and the model will try to predict whether or not the user has Brain Tumor or not.")
st.sidebar.header(":green[What tools where used to make this?]")
st.sidebar.text("The Model was made using a dataset from BraTS-22 along with using Deep Learning to train the model. We made use of Tensorflow, Keras as well as some other Python Libraries to make this complete project. To deply it on web, we used ngrok and Streamlit!")


def import_and_predict(flair,ce,t,model):

    data = get_img_data(flair, ce, t)
    prediction = model.predict(data, verbose=0)

    core = prediction[:, :, :, 1]

    f,ax=get_predicted_imgs(flair,core)
    return f,ax

def read_image(inputImage):
    string_id = inputImage[0:15]
    case_path = os.path.join(root_dir,string_id,inputImage)
    imageReader = vtk.vtkNIFTIImageReader()
    imageReader.SetFileName(case_path)
    imageReader.Update()
    dims = imageReader.GetOutput().GetDimensions()
    data = vtk_to_numpy(imageReader.GetOutput().GetPointData().GetScalars())
    component = imageReader.GetNumberOfScalarComponents()
    if component == 1:
        numpy_data = data.reshape(dims[2], dims[1], dims[0])
        numpy_data = numpy_data.transpose(2, 1, 0)
    elif component == 3 or component == 4:
        if dims[2] == 1:  # a 2D RGB image
            numpy_data = data.reshape(dims[1], dims[0], component)
            numpy_data = numpy_data.transpose(0, 1, 2)
            numpy_data = np.flipud(numpy_data)
        else:
            raise RuntimeError('unknown type')

    return numpy_data

file_flair = st.file_uploader("Please upload your MRI scan file- FLAIR",type=['nii.gz'])
file_t1 = st.file_uploader("Please upload your MRI scan file- T1Ce",type=['nii.gz'])
file_t2 = st.file_uploader("Please upload your MRI scan file- T2",type=['nii.gz'])

st.write("---")
if st.button('Show prediction:'):
    if file_flair is not None:
        flair = read_image(file_flair.name)
    if file_t1 is not None:
        ce = read_image(file_t1.name)
    if file_t2 is not None:
        t = read_image(file_t2.name)

    f,ax = import_and_predict(flair,ce,t,model)
    string_id = file_flair.name[0:15]
    st.markdown(f"**ID:** :red[{string_id}]")
    st.markdown("**Accepted File Type**: :red[NiFti (nii.gz)]")
    st.pyplot(f,use_container_width=True)
    st.markdown("The marked or highlighted region in the left side depicts the detected core of the tumor by the model for the given MRI scan."
             "If there is no significant mark, then that means there is :gray[NO TUMOR]")

