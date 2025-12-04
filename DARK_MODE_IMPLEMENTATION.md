# Dark Mode & Light Mode Implementation

## Overview

The attendance system now supports both light and dark themes, providing users with a comfortable viewing experience in any lighting condition. The theme switcher is easily accessible from the navigation bar and persists user preferences.

## Features

### 🌓 Theme Toggle Button

- **Location**: Navigation bar, next to the user dropdown
- **Icon**: Moon icon (🌙) for light mode, Sun icon (☀️) for dark mode
- **Interaction**: Single click to toggle between themes
- **Animation**: Smooth icon rotation on hover

### 💾 Theme Persistence

- User's theme preference is saved in localStorage
- Theme persists across browser sessions
- Automatic theme restoration on page load

### 🎨 System Preference Detection

- Automatically detects system theme preference (prefers-color-scheme)
- Applies dark mode if system is set to dark (when no manual preference exists)
- Listens for system theme changes and updates accordingly

### ⚡ Smooth Transitions

- Smooth color transitions when switching themes (0.3s ease)
- No jarring color changes
- Professional fade effect between themes

## Technical Implementation

### CSS Variables

Dark mode uses CSS custom properties for easy theme management:

```css
:root {
  /* Light Mode (Default) */
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --text-primary: #111827;
  --text-secondary: #6b7280;
  --border-color: #e5e7eb;
}

[data-theme="dark"] {
  /* Dark Mode */
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --border-color: #334155;
}
```

### Components with Dark Mode Support

#### ✅ Fully Supported Components

1. **Navigation Bar**
   - Theme toggle button with hover effects
   - Dropdown menus with dark backgrounds
2. **Cards**

   - Background colors adapt to theme
   - Border colors adjust automatically
   - Hover effects work in both themes

3. **Forms**

   - Input fields with proper contrast
   - Placeholders with readable colors
   - Labels and helper text adapt

4. **Tables**

   - Header backgrounds
   - Row hover states
   - Striped rows
   - Border colors

5. **Modals**

   - Header, body, and footer backgrounds
   - Close button inverted in dark mode
   - Proper text contrast

6. **Buttons**

   - All button variants (primary, secondary, success, danger, etc.)
   - Maintained gradients and hover effects

7. **Alerts**

   - Success, danger, warning, and info alerts
   - Maintained border-left accents
   - Readable text in both themes

8. **Typography**
   - Headings (h1-h6)
   - Paragraph text
   - Links and inline elements

### JavaScript Implementation

**File**: `static/js/theme.js`

```javascript
// Key functions:
-getCurrentTheme() - // Get current theme from localStorage or system
  applyTheme(theme) - // Apply theme to document
  toggleTheme() - // Toggle between light and dark
  initTheme(); // Initialize theme on page load
```

**Features**:

- IIFE (Immediately Invoked Function Expression) to avoid global scope pollution
- Event listeners for theme toggle button
- System preference change listener
- localStorage management

### HTML Template Changes

**File**: `templates/components/navbar.html`

Added theme toggle button before user dropdown:

```html
<li class="nav-item me-3 d-flex align-items-center">
  <button
    class="btn btn-link nav-link p-2 theme-toggle"
    id="themeToggle"
    type="button"
    aria-label="Toggle theme"
  >
    <i class="fas fa-moon" id="themeIcon"></i>
  </button>
</li>
```

**File**: `templates/base.html`

Added theme.js script before main.js:

```html
<script src="{{ url_for('static', filename='js/theme.js') }}"></script>
```

## Color Palette

### Light Mode Colors

- **Primary Background**: #ffffff (White)
- **Secondary Background**: #f8fafc (Very Light Blue-Gray)
- **Primary Text**: #111827 (Near Black)
- **Secondary Text**: #6b7280 (Medium Gray)
- **Border**: #e5e7eb (Light Gray)

### Dark Mode Colors

- **Primary Background**: #0f172a (Deep Navy)
- **Secondary Background**: #1e293b (Dark Slate)
- **Primary Text**: #f1f5f9 (Off White)
- **Secondary Text**: #94a3b8 (Light Gray-Blue)
- **Border**: #334155 (Medium Slate)

### Accent Colors (Same in Both Modes)

- **Primary**: #4f46e5 (Indigo)
- **Success**: #10b981 (Green)
- **Danger**: #ef4444 (Red)
- **Warning**: #f59e0b (Amber)
- **Info**: #06b6d4 (Cyan)

## Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Accessibility

- **ARIA Labels**: Theme toggle button has proper aria-label
- **Keyboard Navigation**: Toggle button fully keyboard accessible
- **Focus States**: Visible focus outline on toggle button
- **Color Contrast**: WCAG AA compliant in both themes
- **Screen Readers**: Announces "Toggle theme" button

## User Experience

1. **First Visit**:
   - System theme preference detected
   - Dark mode applied if system is dark
2. **Manual Toggle**:
   - Click moon icon to switch to dark mode
   - Click sun icon to switch to light mode
   - Preference saved immediately
3. **Return Visit**:
   - Last selected theme restored automatically
   - No flash of wrong theme (FOUT prevention)

## Performance

- **Lightweight**: theme.js is only ~3KB
- **No Dependencies**: Pure vanilla JavaScript
- **Minimal Repaints**: CSS transitions handle visual changes
- **Cached Preference**: localStorage prevents repeated checks

## Future Enhancements

- [ ] Auto theme switching based on time of day
- [ ] High contrast mode option
- [ ] Custom color theme builder
- [ ] Theme preview before applying
- [ ] Keyboard shortcut (Ctrl+Shift+D) for quick toggle

## Testing Checklist

- [x] Theme toggle button appears in navbar
- [x] Moon icon shows in light mode
- [x] Sun icon shows in dark mode
- [x] Theme persists after page refresh
- [x] All cards display correctly in both themes
- [x] Forms are readable in both themes
- [x] Tables style properly in both themes
- [x] Modals work in both themes
- [x] Alerts are visible in both themes
- [x] Buttons maintain styling in both themes
- [x] System preference detection works
- [x] No console errors
- [x] Mobile responsive

## File Changes Summary

### New Files

- `static/js/theme.js` - Theme switcher logic

### Modified Files

1. `static/css/main.css`

   - Added dark mode CSS variables
   - Updated all components for theme support
   - Added theme toggle button styling

2. `templates/components/navbar.html`

   - Added theme toggle button

3. `templates/base.html`
   - Added theme.js script reference

## Usage for Developers

### Adding New Components

When creating new components, use CSS variables for colors:

```css
.my-component {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}
```

### Custom Dark Mode Styles

If a component needs specific dark mode styling:

```css
[data-theme="dark"] .my-component {
  /* Dark mode specific styles */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
}
```

### Accessing Current Theme in JavaScript

```javascript
const currentTheme = document.documentElement.getAttribute("data-theme");
// Returns 'dark' or null (light mode)
```

## Troubleshooting

### Theme doesn't persist

- Check browser localStorage is enabled
- Verify no browser extensions blocking localStorage

### Icon doesn't change

- Ensure Font Awesome is loaded
- Check console for JavaScript errors

### Colors look wrong

- Clear browser cache
- Verify main.css is loaded properly
- Check CSS variables are defined

## Credits

- **Design System**: Tailwind CSS inspired colors
- **Icons**: Font Awesome 6
- **Implementation**: Custom vanilla JavaScript

---

**Date Implemented**: 2024
**Version**: 1.0
**Status**: ✅ Production Ready
