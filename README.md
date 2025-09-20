# AgeColorValidator

A simple **Java Servlet 5.0** web application built with **Eclipse IDE 2025-09** and deployed on **Apache Tomcat 10.1**. The project validates user input (first name, last name, age, and favorite color) and displays a personalized message.

## 🛠️ Tools & Technologies Used

* **Java 17+** (compatible with Jakarta Servlet 5.0)
* **Eclipse IDE 2025-09**
* **Apache Tomcat 10.1**
* **Jakarta Servlet API 5.0**
* **HTML & JSP** for UI and result rendering
* **Git & GitHub** for version control

## 📂 Project Structure

```
AgeColorValidator/
 ├─ src/main/java/com/assignment/AgeColorServlet.java
 ├─ src/main/webapp/
 │   ├─ index.html
 │   ├─ result.jsp
 │   └─ WEB-INF/web.xml
 ├─ screenshots/ (screenshots of form and output)
 └─ README.md
```

## 🚀 Features

### 1. **Form (index.html)**
   * First Name (required, must be ≥ 2 characters)
   * Last Name (required, must be ≥ 2 characters)
   * Age (numeric, ≥ 0)
   * Favorite Color (dropdown: Red, Blue, Green, Yellow)

### 2. **Validation in Servlet**
   * If name fields are empty or less than 2 letters → error message
   * If age < 18 → `"Hello [FirstName] [LastName], you are still a minor."`
   * If age ≥ 18 → `"Hello [FirstName] [LastName], you are adult."`
   * Displays favorite color message

### 3. **Result Display**
   * The servlet forwards the response to `result.jsp` using `RequestDispatcher`.

## ⚙️ How to Run

### 1️⃣ Import Project into Eclipse
1. Open **Eclipse IDE 2025-09**
2. Go to: `File → Import → Existing Projects into Workspace`
3. Select the project folder: `C:\Users\Hey\eclipse-workspace\AgeColorValidator`

### 2️⃣ Configure Tomcat
1. In Eclipse, open **Servers view**
2. Add **Apache Tomcat v10.1**
3. Set **Dynamic Web Module version = 5.0**
4. Deploy the project to Tomcat

### 3️⃣ Run Application
1. Right-click the project → `Run As → Run on Server`
2. Select **Tomcat 10.1**
3. Eclipse will start Tomcat and deploy the app
4. Open browser and visit: 👉 `http://localhost:8080/AgeColorValidator/`

### 4️⃣ Test Input
* Enter **First Name, Last Name, Age, Favorite Color**
* Click **Submit**
* The servlet will validate and forward result to `result.jsp`

## 📸 Screenshots & Explanations

### 📝 index.html (Form Page)

![Index Page](screenshots/index.png)

This is the main page where the user enters their **first name**, **last name**, **age**, and selects a **favorite color**. Clicking **Submit** sends the form data to `AgeColorServlet`.

### ⚙️ Servlet Processing

![Servlet Processing](screenshots/servlet-processing.png)

The form sends the data to **AgeColorServlet.java**, which processes and forwards it to `result.jsp` for display.

### 🖥️ result.jsp (Result Page)

![Result Page](screenshots/result.png)

This page displays the processed user information, showing their name, age, and selected color after the servlet validates the data.

### 🖼️ Additional Screenshots

![Eclipse Workspace](screenshots/eclipse-workspace.png)
*Eclipse workspace where the project was built and run.*

![Initial Test](screenshots/initial-test.png)
*Initial test of the AgeColorValidator form.*

## 📌 GitHub Workflow

```bash
# Initialize git repo
git init

# Add remote
git remote add origin https://github.com/ishimweegide23/AgeColorValidator.git

# Stage and commit
git add .
git commit -m "Initial commit - AgeColorValidator project"

# Push to GitHub
git branch -M main
git push -u origin main
```

## 👥 Collaborators

* **Owner:** @ishimweegide23
* **Collaborators Added:**
   * `kardara` (Instructor)
   * `dushimimanapatrick@gmail.com` (Teacher Patrick)
