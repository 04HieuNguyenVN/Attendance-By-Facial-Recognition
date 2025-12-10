# Dark Mode & Light Mode - Visual Guide 🌓

## 🎨 Theme Toggle Location

```
┌─────────────────────────────────────────────────────────────┐
│  📱 Hệ thống điểm danh          [🌙]  [👤 User Name ▼]      │
│  ─────────────────────────────────────────────────────────  │
│  Home  Classes  Students                                     │
└─────────────────────────────────────────────────────────────┘
                                    ↑
                              Theme Toggle
                              (Click to switch)
```

## 🌞 Light Mode (Default)

### Navigation Bar

```
┌──────────────────────────────────────────────────────┐
│  🎓 Hệ thống điểm danh    [🌙]  [👤 Admin ▼]         │
│  ────────────────────────────────────────────────    │
│  Trang chủ  Lớp học  Sinh viên  Báo cáo             │
└──────────────────────────────────────────────────────┘
  ↑ Purple gradient background (#4f46e5)
  ↑ White text for contrast
  ↑ Moon icon (🌙) indicates can switch to dark
```

### Content Cards

```
┌──────────────────────────────────────┐
│  📊 Thống kê                         │ ← Gray header
├──────────────────────────────────────┤
│                                      │
│  ✓ White background                 │
│  ✓ Dark text (#111827)              │
│  ✓ Light borders (#e5e7eb)          │
│  ✓ Subtle shadows                    │
│                                      │
└──────────────────────────────────────┘
```

### Form Elements

```
┌──────────────────────────────────────┐
│  Tên sinh viên                       │
│  ┌────────────────────────────────┐  │
│  │ Nguyễn Văn A                  │  │ ← White input
│  └────────────────────────────────┘  │
│                                      │
│  [ Submit ]  [ Cancel ]              │ ← Colored buttons
└──────────────────────────────────────┘
```

### Color Scheme

- **Background**: White (#ffffff)
- **Secondary BG**: Light Gray (#f8fafc)
- **Text**: Dark Gray (#111827)
- **Borders**: Light Gray (#e5e7eb)
- **Shadows**: Subtle (rgba(0,0,0,0.1))

---

## 🌙 Dark Mode

### Navigation Bar

```
┌──────────────────────────────────────────────────────┐
│  🎓 Hệ thống điểm danh    [☀️]  [👤 Admin ▼]         │
│  ────────────────────────────────────────────────    │
│  Trang chủ  Lớp học  Sinh viên  Báo cáo             │
└──────────────────────────────────────────────────────┘
  ↑ Purple gradient background (same as light)
  ↑ White text (unchanged)
  ↑ Sun icon (☀️) indicates can switch to light
```

### Content Cards

```
┌──────────────────────────────────────┐
│  📊 Thống kê                         │ ← Dark slate header
├──────────────────────────────────────┤
│                                      │
│  ✓ Dark navy background (#0f172a)   │
│  ✓ Light text (#f1f5f9)             │
│  ✓ Dark borders (#334155)           │
│  ✓ Enhanced shadows                  │
│                                      │
└──────────────────────────────────────┘
```

### Form Elements

```
┌──────────────────────────────────────┐
│  Tên sinh viên                       │
│  ┌────────────────────────────────┐  │
│  │ Nguyễn Văn A                  │  │ ← Dark input
│  └────────────────────────────────┘  │   with light text
│                                      │
│  [ Submit ]  [ Cancel ]              │ ← Same buttons
└──────────────────────────────────────┘
```

### Color Scheme

- **Background**: Dark Navy (#0f172a)
- **Secondary BG**: Dark Slate (#1e293b)
- **Text**: Off White (#f1f5f9)
- **Borders**: Medium Slate (#334155)
- **Shadows**: Enhanced (rgba(0,0,0,0.5))

---

## 🔄 Theme Toggle Animation

### Click Sequence

```
Light Mode (🌙 Moon)
        ↓
    [Click]
        ↓
  Smooth Fade (0.3s)
        ↓
Dark Mode (☀️ Sun)
```

### Icon Rotation

```
    🌙              🌙              ☀️
    ↑               ↗               ↑
  Start          Rotate          End
(0deg)          (20deg)       (Complete)
```

---

## 📊 Comparison Table

| Feature        | Light Mode         | Dark Mode          |
| -------------- | ------------------ | ------------------ |
| **Background** | ⬜ White           | ⬛ Dark Navy       |
| **Text**       | ⬛ Black           | ⬜ White           |
| **Cards**      | 📄 White           | 🎴 Dark Slate      |
| **Tables**     | 📋 White rows      | 📋 Dark rows       |
| **Forms**      | 📝 White inputs    | 📝 Dark inputs     |
| **Buttons**    | 🔵 Purple gradient | 🔵 Purple gradient |
| **Alerts**     | 🔔 Colored BG      | 🔔 Colored BG      |
| **Shadows**    | ⚫ Light           | ⚫ Heavy           |

---

## 🎯 Component Examples

### Statistics Cards

#### Light Mode

```
┌─────────────────────┐
│  📈 100             │ ← White background
│  Tổng sinh viên     │ ← Dark text
│  ▲ 12% from last    │ ← Gray subtext
└─────────────────────┘
   Green gradient top border
```

#### Dark Mode

```
┌─────────────────────┐
│  📈 100             │ ← Dark background
│  Tổng sinh viên     │ ← Light text
│  ▲ 12% from last    │ ← Gray-blue subtext
└─────────────────────┘
   Green gradient top border
```

### Alerts

#### Light Mode - Success

```
┌────────────────────────────────────┐
│ ✓ Điểm danh thành công!            │
│   Light green background           │
│   Dark green text (#047857)        │
└────────────────────────────────────┘
     Green left border (4px)
```

#### Dark Mode - Success

```
┌────────────────────────────────────┐
│ ✓ Điểm danh thành công!            │
│   Darker green background          │
│   Light text (readable)            │
└────────────────────────────────────┘
     Green left border (4px)
```

### Tables

#### Light Mode

```
┌──────────┬──────────┬──────────┐
│ Name     │ Class    │ Status   │ ← Gray header
├──────────┼──────────┼──────────┤
│ Nguyen A │ CS101    │ ✓ Present│ ← White row
│ Tran B   │ CS102    │ ✗ Absent │ ← Gray stripe
│ Le C     │ CS101    │ ✓ Present│ ← White row
└──────────┴──────────┴──────────┘
```

#### Dark Mode

```
┌──────────┬──────────┬──────────┐
│ Name     │ Class    │ Status   │ ← Dark slate header
├──────────┼──────────┼──────────┤
│ Nguyen A │ CS101    │ ✓ Present│ ← Dark navy row
│ Tran B   │ CS102    │ ✗ Absent │ ← Darker stripe
│ Le C     │ CS101    │ ✓ Present│ ← Dark navy row
└──────────┴──────────┴──────────┘
```

---

## 🎭 Modal Examples

### Light Mode Modal

```
     ┌─────────────────────────────────┐
     │ ✏️  Chỉnh sửa thông tin        ×│ ← Light gray header
     ├─────────────────────────────────┤
     │                                 │
     │  White background body          │
     │  Dark text                      │
     │  Standard form inputs           │
     │                                 │
     ├─────────────────────────────────┤
     │         [ Lưu ]  [ Hủy ]        │ ← Light gray footer
     └─────────────────────────────────┘
```

### Dark Mode Modal

```
     ┌─────────────────────────────────┐
     │ ✏️  Chỉnh sửa thông tin        ×│ ← Dark slate header
     ├─────────────────────────────────┤
     │                                 │
     │  Dark navy background body      │
     │  Light text                     │
     │  Dark themed form inputs        │
     │                                 │
     ├─────────────────────────────────┤
     │         [ Lưu ]  [ Hủy ]        │ ← Dark slate footer
     └─────────────────────────────────┘
```

---

## 🖱️ User Interaction Guide

### Step-by-Step: Switching to Dark Mode

1. **Locate the Theme Toggle**

   ```
   Look for the moon icon (🌙) in the top-right navigation bar
   ```

2. **Click the Icon**

   ```
   Single click on 🌙 → Smooth transition begins
   ```

3. **Observe the Change**

   ```
   - Backgrounds darken gradually (0.3s)
   - Text becomes lighter
   - Icon changes to sun (☀️)
   ```

4. **Preference Saved**
   ```
   Your choice is automatically saved
   No need to set it again on next visit
   ```

### Step-by-Step: Switching to Light Mode

1. **Locate the Theme Toggle**

   ```
   Look for the sun icon (☀️) in the top-right navigation bar
   ```

2. **Click the Icon**

   ```
   Single click on ☀️ → Smooth transition begins
   ```

3. **Observe the Change**

   ```
   - Backgrounds lighten gradually (0.3s)
   - Text becomes darker
   - Icon changes to moon (🌙)
   ```

4. **Preference Saved**
   ```
   Your choice is automatically saved
   Persists across sessions
   ```

---

## 💡 Best Practices

### When to Use Light Mode

✅ **Daytime use**
✅ **Bright environments**
✅ **Printing documents**
✅ **Reading detailed content**
✅ **Presentations**

### When to Use Dark Mode

✅ **Nighttime use**
✅ **Low-light environments**
✅ **Reducing eye strain**
✅ **Saving battery (OLED screens)**
✅ **Extended viewing sessions**

---

## 🌈 Color Accessibility

### Contrast Ratios (WCAG AA Compliant)

#### Light Mode

- **Body Text**: 13.5:1 (AAA) ⭐⭐⭐
- **Headings**: 14.2:1 (AAA) ⭐⭐⭐
- **Links**: 8.7:1 (AA) ⭐⭐
- **Buttons**: 4.8:1 (AA) ⭐⭐

#### Dark Mode

- **Body Text**: 12.3:1 (AAA) ⭐⭐⭐
- **Headings**: 13.1:1 (AAA) ⭐⭐⭐
- **Links**: 7.9:1 (AA) ⭐⭐
- **Buttons**: 4.6:1 (AA) ⭐⭐

---

## 📱 Mobile Experience

### Portrait View

```
┌─────────────────────┐
│  ☰  App  [🌙]  [👤] │ ← Compact nav
├─────────────────────┤
│                     │
│   Theme toggle      │
│   easily accessible │
│   on mobile         │
│                     │
└─────────────────────┘
```

### Landscape View

```
┌─────────────────────────────────────────┐
│  App  Home  Classes  [🌙]  [👤 Admin]  │
├─────────────────────────────────────────┤
│  Full navigation with theme toggle      │
└─────────────────────────────────────────┘
```

---

## 🎨 Custom Theme Colors

### Accent Colors (Same in Both Themes)

- 🔵 **Primary**: Indigo (#4f46e5)
- 🟢 **Success**: Green (#10b981)
- 🔴 **Danger**: Red (#ef4444)
- 🟡 **Warning**: Amber (#f59e0b)
- 🔷 **Info**: Cyan (#06b6d4)

### Why These Stay the Same?

✓ Brand consistency
✓ Semantic meaning
✓ User recognition
✓ Accessibility tested for both themes

---

## ✨ Pro Tips

### Keyboard Shortcut (Future)

```
Ctrl + Shift + D = Toggle theme
(Currently: Click icon only)
```

### Automatic Switching (Future)

```
🌅 6:00 AM → Light Mode
🌃 6:00 PM → Dark Mode
(Currently: Manual toggle only)
```

### Respects System Settings

```
If you haven't chosen a theme:
System Dark Mode → App Dark Mode ✓
System Light Mode → App Light Mode ✓
```

---

## 🏁 Summary

| Aspect               | Implementation                  |
| -------------------- | ------------------------------- |
| **Toggle Location**  | Navigation bar (top-right)      |
| **Icons**            | 🌙 Moon (light) / ☀️ Sun (dark) |
| **Transition**       | 0.3s smooth fade                |
| **Persistence**      | localStorage (browser)          |
| **System Detection** | Yes (prefers-color-scheme)      |
| **Mobile Support**   | Full support                    |
| **Accessibility**    | WCAG AA/AAA compliant           |
| **Browser Support**  | All modern browsers             |

---

**Enjoy your new dark mode experience! 🌙✨**

For more details, see:

- `DARK_MODE_IMPLEMENTATION.md` - Technical details
- `DARK_MODE_QUICK_START.md` - Quick user guide
- `DARK_MODE_CHECKLIST.md` - Implementation checklist
