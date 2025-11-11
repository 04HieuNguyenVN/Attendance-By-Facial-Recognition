# Components Documentation

Thư mục này chứa các component có thể tái sử dụng cho hệ thống điểm danh.

## Danh sách Components

### 1. navbar.html

**Navigation bar chung cho toàn bộ hệ thống**

Sử dụng:

```jinja2
{% set active_page = 'home' %}  {# hoặc 'students', 'reports', 'classes', 'settings' #}
{% include 'components/navbar.html' %}
```

### 2. footer.html

**Footer chung cho toàn bộ hệ thống**

Sử dụng:

```jinja2
{% include 'components/footer.html' %}
```

### 3. scripts.html

**Các script chung (Bootstrap, jQuery, Chart.js, utility functions)**

Sử dụng:

```jinja2
{% include 'components/scripts.html' %}
```

Utility functions có sẵn:

- `showToast(message, type)` - Hiển thị toast notification
- `showLoading(show)` - Hiển thị/ẩn loading spinner
- `formatDate(dateString)` - Format ngày theo định dạng VN
- `formatDateTime(dateString)` - Format ngày giờ theo định dạng VN

### 4. page_header.html

**Header tiêu đề trang với icon và subtitle**

Sử dụng:

```jinja2
{% set page_title = "Quản lý sinh viên" %}
{% set page_subtitle = "Thêm, sửa, xóa sinh viên" %}
{% set page_icon = "fas fa-users" %}
{% include 'components/page_header.html' %}
```

### 5. loading_spinner.html

**Loading spinner animation**

Sử dụng:

```jinja2
{% set loading_text = "Đang tải dữ liệu..." %}  {# optional #}
{% include 'components/loading_spinner.html' %}
```

### 6. alert.html

**Alert/notification component**

Sử dụng:

```jinja2
{% set alert_type = "success" %}  {# success/danger/warning/info #}
{% set alert_message = "Lưu thành công!" %}
{% include 'components/alert.html' %}
```

### 7. delete_modal.html

**Modal xác nhận xóa**

Sử dụng:

```jinja2
{# Include trong page #}
{% include 'components/delete_modal.html' %}

{# Gọi từ JavaScript #}
<script>
    function deleteStudent(id, name) {
        openDeleteModal(id, name, function(studentId) {
            // Your delete logic here
            fetch(`/api/students/${studentId}`, {method: 'DELETE'})
                .then(response => response.json())
                .then(data => {
                    showToast('Xóa thành công!', 'success');
                    location.reload();
                });
        });
    }
</script>
```

### 8. search_filter.html

**Search box và filter dropdown**

Sử dụng:

```jinja2
{% set search_placeholder = "Tìm kiếm sinh viên..." %}
{% set filter_options = [
    {'value': 'cntt', 'label': 'CNTT'},
    {'value': 'kt', 'label': 'Kế toán'}
] %}
{% include 'components/search_filter.html' %}
```

### 9. stat_card.html

**Card hiển thị thống kê**

Sử dụng:

```jinja2
{% set stat_icon = "fas fa-users" %}
{% set stat_value = "150" %}
{% set stat_label = "Tổng sinh viên" %}
{% set stat_color = "primary" %}  {# optional #}
{% set stat_trend = 12 %}  {# optional, số dương/âm cho trend #}
{% include 'components/stat_card.html' %}
```

### 10. empty_state.html

**Hiển thị khi không có dữ liệu**

Sử dụng:

```jinja2
{% set empty_icon = "fas fa-users" %}
{% set empty_title = "Chưa có sinh viên" %}
{% set empty_message = "Hãy thêm sinh viên đầu tiên" %}
{% set empty_action_text = "Thêm sinh viên" %}  {# optional #}
{% set empty_action_link = "#addStudentModal" %}  {# optional #}
{% include 'components/empty_state.html' %}
```

## Cách sử dụng Base Template

File `base.html` đã được cấu trúc lại để sử dụng các component trên.

### Extend base template:

```jinja2
{% extends 'base.html' %}
{% set active_page = 'students' %}

{% block title %}Quản lý sinh viên{% endblock %}

{% block extra_css %}
<style>
    /* Custom CSS cho trang này */
</style>
{% endblock %}

{% block content %}
    <!-- Nội dung trang -->
    <h1>Danh sách sinh viên</h1>

    {% include 'components/search_filter.html' %}

    <!-- Your content here -->
{% endblock %}

{% block extra_js %}
<script>
    // Custom JavaScript cho trang này
</script>
{% endblock %}
```

## Best Practices

1. **Luôn set active_page** khi extend base.html để highlight menu đúng
2. **Sử dụng component thay vì copy-paste** code để dễ maintain
3. **Truyền đủ parameters** khi include component
4. **Test responsive** khi thêm component mới
5. **Document** các parameters mới nếu customize component

## Ví dụ hoàn chỉnh

Xem file `students.html`, `reports.html`, `classes.html` để tham khảo cách sử dụng.
