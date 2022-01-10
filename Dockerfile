#Base Image to use
FROM python:3.7.9-slim

#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y libssl-dev

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
# RUN pip install -r app/requirements.txt

RUN pip install plotly==4.14.3
RUN pip install pandas==1.2.3
RUN pip install streamlit==0.78.0
RUN pip install omegaconf==2.0.6
RUN pip install numpy==1.20.1
RUN pip install altair==4.1.0
RUN pip install scikit_learn==0.24.1
RUN pip install shap==0.40.0
RUN pip install xgboost==1.3.3
RUN pip install matplotlib==3.3.2

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--logger.level=error"]
