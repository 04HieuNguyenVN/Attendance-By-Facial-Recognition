/**
 * Theme Switcher - Dark/Light Mode Toggle
 * Handles theme switching with localStorage persistence
 */

(function () {
  "use strict";

  const THEME_KEY = "attendance-theme";
  const THEME_DARK = "dark";
  const THEME_LIGHT = "light";

  // Get DOM elements
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const html = document.documentElement;

  /**
   * Get the current theme from localStorage or system preference
   * @returns {string} The current theme ('dark' or 'light')
   */
  function getCurrentTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY);
    if (savedTheme) {
      return savedTheme;
    }

    // Check system preference
    if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      return THEME_DARK;
    }

    return THEME_LIGHT;
  }

  /**
   * Apply the theme to the document
   * @param {string} theme - The theme to apply ('dark' or 'light')
   */
  function applyTheme(theme) {
    if (theme === THEME_DARK) {
      html.setAttribute("data-theme", THEME_DARK);
      if (themeIcon) {
        themeIcon.classList.remove("fa-moon");
        themeIcon.classList.add("fa-sun");
      }
    } else {
      html.removeAttribute("data-theme");
      if (themeIcon) {
        themeIcon.classList.remove("fa-sun");
        themeIcon.classList.add("fa-moon");
      }
    }

    localStorage.setItem(THEME_KEY, theme);
  }

  /**
   * Toggle between dark and light themes
   */
  function toggleTheme() {
    const currentTheme = getCurrentTheme();
    const newTheme = currentTheme === THEME_DARK ? THEME_LIGHT : THEME_DARK;
    applyTheme(newTheme);
  }

  /**
   * Initialize the theme on page load
   */
  function initTheme() {
    const theme = getCurrentTheme();
    applyTheme(theme);
  }

  // Initialize theme when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTheme);
  } else {
    initTheme();
  }

  // Add event listener to toggle button
  if (themeToggle) {
    themeToggle.addEventListener("click", toggleTheme);
  }

  // Listen for system theme changes
  if (window.matchMedia) {
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", (e) => {
        // Only auto-switch if user hasn't manually set a preference
        if (!localStorage.getItem(THEME_KEY)) {
          applyTheme(e.matches ? THEME_DARK : THEME_LIGHT);
        }
      });
  }
})();
