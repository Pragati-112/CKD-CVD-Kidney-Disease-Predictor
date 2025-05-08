// script.js
document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const signupForm = document.getElementById("signupForm");
    const loginTab = document.getElementById("loginTab");
    const signupTab = document.getElementById("signupTab");
    const switchToSignup = document.getElementById("switchToSignup");
    const switchToLogin = document.getElementById("switchToLogin");
  
    // Switch to Sign-Up Form
    const showSignupForm = () => {
      loginForm.classList.remove("active-form");
      signupForm.classList.add("active-form");
      loginTab.classList.remove("active-tab");
      signupTab.classList.add("active-tab");
    };
  
    // Switch to Login Form
    const showLoginForm = () => {
      signupForm.classList.remove("active-form");
      loginForm.classList.add("active-form");
      signupTab.classList.remove("active-tab");
      loginTab.classList.add("active-tab");
    };
  
    loginTab.addEventListener("click", showLoginForm);
    signupTab.addEventListener("click", showSignupForm);
    switchToSignup.addEventListener("click", (e) => {
      e.preventDefault();
      showSignupForm();
    });
    switchToLogin.addEventListener("click", (e) => {
      e.preventDefault();
      showLoginForm();
    });
  });
  