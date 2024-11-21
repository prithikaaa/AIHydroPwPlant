# **HydroStorm**

---

## **Overview**

**HydroStorm** is a web platform that leverages **machine learning (ML)** to revolutionize the planning and design of hydropower plants along rivers. It uses advanced **satellite image processing** to analyze map patterns and identify potential construction sites, offering actionable insights for civil engineers and architects.
Article: https://medium.com/@hydrostorm1000/hydrostorm-d685c4fe4bd6

---

## **Key Features**

### **1. Site Identification**
- Analyzes **satellite images** to identify optimal locations for hydropower plant construction.
- Maps dam statuses using **colored dots**, distinguishing between:
  - **Proposed**
  - **Completed**
  - **Under Construction**

### **2. Hydropower Plant Type Prediction**
- Predicts plant types such as:
  - **Storage**
  - **Pumped Storage**
  - **Run-of-River**
- Utilizes **logistic regression** trained on real-world datasets for accurate forecasting.

### **3. Dam and Reservoir Prediction**
- Employs **ML models** like:
  - **Random Forest**
  - **CT-GAN**
  - **Multioutput Regressor**
- Predicts parameters like:
  - **Dam height**
  - **Reservoir capacity**
  - **Reservoir area**
- Uses **eight distinct inputs**, including:
  - **Water head height**
  - **Wave energy**

### **4. Real-World Data Integration**
- Integrates datasets from **India's hydropower projects**.
- Collects data through **remote sensors** to ensure accuracy and scalability.

---

## **Technologies Used**

### **Backend**
- **Flask** for server-side logic and API development.

### **Frontend**
- **HTML**, **CSS**, and **JavaScript** for a responsive and user-friendly interface.

### **Machine Learning**
- **Logistic Regression**, **Random Forest**, **CT-GAN**, and **Multioutput Regressor** for predictive modeling.

### **Data Collection**
- Utilizes **remote sensing technology** for satellite image analysis and dataset generation.

---

## **Impact and Goals**
HydroStorm aims to:
- Enhance **project planning** and **sustainability** in renewable energy development.
- Optimize resources for **civil engineers** and **architects**.
- Foster **collaborative innovation** in hydropower infrastructure design.

---

## **Setup and Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/prithikaaa/AIHydroPwPlant.git
   cd HydroStorm
