# Codevsi
## Accessing the App on Web :

## Installing the Web app as executable on the machine :
- First of all, you need to install **npm** (Node package manager) and by extension **Node.js** on your machine. On Windows, you can install the packages From the [Node.js Downloads](https://nodejs.org/en/download/) page. To do so on MacOS/Linux, run this command in the terminal :
```
npm install -g npm
```
- Check **npm** and **Node.js** installations using these commands :
```
node -v
npm -v
``` 
- Install **nativefier** by running the below command:
```
npm install -g nativefier
```
- Go to the directory where you want to install the executable and run this command depending on your machine (Windows/Linux/MacOS):
```
nativefier --name 'Codevsi' 'https://codevsiwebapp.streamlit.app/' --platform 'windows'
```
```
nativefier --name 'Codevsi' 'https://codevsiwebapp.streamlit.app/' --platform 'linux'
```
```
nativefier --name 'Codevsi' 'https://codevsiwebapp.streamlit.app/' --platform 'mac'
```
This should create the executable in the chosen directory.
