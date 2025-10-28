/**
 * Hệ thống điểm danh - JavaScript chính
 * Quản lý các chức năng chung của ứng dụng
 */

// ========================================
// GLOBAL VARIABLES
// ========================================
const App = {
    config: {
        apiBaseUrl: '/api',
        refreshInterval: 5000,
        animationDuration: 300
    },
    
    state: {
        isLoading: false,
        currentUser: null,
        notifications: []
    },
    
    elements: {
        loadingSpinner: null,
        notificationContainer: null
    }
};

// ========================================
// UTILITY FUNCTIONS
// ========================================
const Utils = {
    /**
     * Format date to Vietnamese format
     */
    formatDate(date) {
        if (!date) return '';
        const d = new Date(date);
        return d.toLocaleDateString('vi-VN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    },

    /**
     * Format time ago
     */
    timeAgo(date) {
        if (!date) return '';
        const now = new Date();
        const past = new Date(date);
        const diffInSeconds = Math.floor((now - past) / 1000);
        
        if (diffInSeconds < 60) return 'Vừa xong';
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} phút trước`;
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} giờ trước`;
        return `${Math.floor(diffInSeconds / 86400)} ngày trước`;
    },

    /**
     * Show loading spinner
     */
    showLoading(element = null) {
        App.state.isLoading = true;
        if (element) {
            element.innerHTML = '<div class="loading-container"><div class="spinner-custom"></div></div>';
        }
    },

    /**
     * Hide loading spinner
     */
    hideLoading() {
        App.state.isLoading = false;
    },

    /**
     * Show notification
     */
    showNotification(message, type = 'info', duration = 5000) {
        const notification = {
            id: Date.now(),
            message,
            type,
            duration
        };
        
        App.state.notifications.push(notification);
        this.renderNotifications();
        
        // Auto remove after duration
        setTimeout(() => {
            this.removeNotification(notification.id);
        }, duration);
    },

    /**
     * Remove notification
     */
    removeNotification(id) {
        App.state.notifications = App.state.notifications.filter(n => n.id !== id);
        this.renderNotifications();
    },

    /**
     * Render notifications
     */
    renderNotifications() {
        const container = document.getElementById('notification-container');
        if (!container) return;

        container.innerHTML = App.state.notifications.map(notification => `
            <div class="alert alert-${notification.type}-custom alert-dismissible fade show" role="alert">
                <i class="fas fa-${this.getNotificationIcon(notification.type)} me-2"></i>
                ${notification.message}
                <button type="button" class="btn-close" onclick="Utils.removeNotification(${notification.id})"></button>
            </div>
        `).join('');
    },

    /**
     * Get notification icon
     */
    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            danger: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Throttle function
     */
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

// ========================================
// API FUNCTIONS
// ========================================
const API = {
    /**
     * Make API request
     */
    async request(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const config = { ...defaultOptions, ...options };
        
        try {
            Utils.showLoading();
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'Có lỗi xảy ra');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            Utils.showNotification(error.message, 'danger');
            throw error;
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * GET request
     */
    async get(url) {
        return this.request(url, { method: 'GET' });
    },

    /**
     * POST request
     */
    async post(url, data) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * PUT request
     */
    async put(url, data) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    /**
     * DELETE request
     */
    async delete(url) {
        return this.request(url, { method: 'DELETE' });
    },

    /**
     * Upload file
     */
    async upload(url, formData) {
        return this.request(url, {
            method: 'POST',
            body: formData,
            headers: {} // Let browser set Content-Type for FormData
        });
    }
};

// ========================================
// FORM HANDLING
// ========================================
const FormHandler = {
    /**
     * Initialize form validation
     */
    init() {
        const forms = document.querySelectorAll('.needs-validation');
        forms.forEach(form => {
            form.addEventListener('submit', this.handleSubmit.bind(this));
        });
    },

    /**
     * Handle form submission
     */
    handleSubmit(event) {
        const form = event.target;
        if (!form.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        }
        form.classList.add('was-validated');
    },

    /**
     * Reset form
     */
    reset(form) {
        form.reset();
        form.classList.remove('was-validated');
        // Clear custom validation
        const inputs = form.querySelectorAll('.form-control-custom');
        inputs.forEach(input => {
            input.classList.remove('is-valid', 'is-invalid');
        });
    },

    /**
     * Validate field
     */
    validateField(field) {
        const value = field.value.trim();
        const type = field.type;
        const required = field.hasAttribute('required');
        
        let isValid = true;
        let message = '';

        if (required && !value) {
            isValid = false;
            message = 'Trường này là bắt buộc';
        } else if (type === 'email' && value && !this.isValidEmail(value)) {
            isValid = false;
            message = 'Email không hợp lệ';
        } else if (type === 'tel' && value && !this.isValidPhone(value)) {
            isValid = false;
            message = 'Số điện thoại không hợp lệ';
        }

        this.setFieldValidation(field, isValid, message);
        return isValid;
    },

    /**
     * Check if email is valid
     */
    isValidEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },

    /**
     * Check if phone is valid
     */
    isValidPhone(phone) {
        const re = /^[0-9]{10,11}$/;
        return re.test(phone.replace(/\s/g, ''));
    },

    /**
     * Set field validation state
     */
    setFieldValidation(field, isValid, message) {
        field.classList.remove('is-valid', 'is-invalid');
        field.classList.add(isValid ? 'is-valid' : 'is-invalid');
        
        // Remove existing feedback
        const existingFeedback = field.parentNode.querySelector('.invalid-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }
        
        // Add feedback if invalid
        if (!isValid && message) {
            const feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            feedback.textContent = message;
            field.parentNode.appendChild(feedback);
        }
    }
};

// ========================================
// MODAL HANDLING
// ========================================
const ModalHandler = {
    /**
     * Show modal
     */
    show(modalId) {
        const modal = new bootstrap.Modal(document.getElementById(modalId));
        modal.show();
    },

    /**
     * Hide modal
     */
    hide(modalId) {
        const modal = bootstrap.Modal.getInstance(document.getElementById(modalId));
        if (modal) {
            modal.hide();
        }
    },

    /**
     * Initialize modal events
     */
    init() {
        // Auto-hide modals after form submission
        document.addEventListener('submit', (e) => {
            const form = e.target;
            if (form.closest('.modal')) {
                setTimeout(() => {
                    const modal = form.closest('.modal');
                    const modalInstance = bootstrap.Modal.getInstance(modal);
                    if (modalInstance) {
                        modalInstance.hide();
                    }
                }, 1000);
            }
        });
    }
};

// ========================================
// TABLE HANDLING
// ========================================
const TableHandler = {
    /**
     * Initialize data tables
     */
    init() {
        const tables = document.querySelectorAll('.table-custom');
        tables.forEach(table => {
            this.addSorting(table);
            this.addFiltering(table);
        });
    },

    /**
     * Add sorting to table
     */
    addSorting(table) {
        const headers = table.querySelectorAll('th[data-sort]');
        headers.forEach(header => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                this.sortTable(table, header.dataset.sort);
            });
        });
    },

    /**
     * Sort table
     */
    sortTable(table, column) {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const isAscending = table.dataset.sortDirection !== 'asc';
        
        rows.sort((a, b) => {
            const aVal = a.querySelector(`[data-sort-value="${column}"]`)?.textContent || '';
            const bVal = b.querySelector(`[data-sort-value="${column}"]`)?.textContent || '';
            
            if (isAscending) {
                return aVal.localeCompare(bVal);
            } else {
                return bVal.localeCompare(aVal);
            }
        });
        
        rows.forEach(row => tbody.appendChild(row));
        table.dataset.sortDirection = isAscending ? 'asc' : 'desc';
    },

    /**
     * Add filtering to table
     */
    addFiltering(table) {
        const filterInput = table.parentNode.querySelector('.table-filter');
        if (filterInput) {
            filterInput.addEventListener('input', Utils.debounce((e) => {
                this.filterTable(table, e.target.value);
            }, 300));
        }
    },

    /**
     * Filter table
     */
    filterTable(table, searchTerm) {
        const tbody = table.querySelector('tbody');
        const rows = tbody.querySelectorAll('tr');
        
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            const matches = text.includes(searchTerm.toLowerCase());
            row.style.display = matches ? '' : 'none';
        });
    }
};

// ========================================
// INITIALIZATION
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all handlers
    FormHandler.init();
    ModalHandler.init();
    TableHandler.init();
    
    // Initialize notification container
    App.elements.notificationContainer = document.getElementById('notification-container');
    
    // Auto-refresh data
    if (App.config.refreshInterval > 0) {
        setInterval(() => {
            // Refresh attendance data if on main page
            if (window.location.pathname === '/') {
                refreshAttendanceData();
            }
        }, App.config.refreshInterval);
    }
    
    console.log('Hệ thống điểm danh đã khởi tạo thành công');
});

// ========================================
// GLOBAL FUNCTIONS (for HTML onclick)
// ========================================
window.refreshAttendanceData = async function() {
    try {
        const response = await API.get('/api/attendance/today');
        if (response.success) {
            updateAttendanceList(response.data);
        }
    } catch (error) {
        console.error('Error refreshing attendance data:', error);
    }
};

window.updateAttendanceList = function(attendanceData) {
    const container = document.getElementById('attendance-list');
    if (!container) return;
    
    if (attendanceData.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-4">Chưa có sinh viên nào điểm danh</div>';
        return;
    }
    
    container.innerHTML = attendanceData.map(item => `
        <div class="attendance-item-custom">
            <div class="student-info">
                <div>
                    <div class="student-name">${item.name}</div>
                    <div class="attendance-time">${Utils.timeAgo(item.timestamp)}</div>
                </div>
                <div class="student-id">${item.student_id || 'N/A'}</div>
            </div>
        </div>
    `).join('');
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { App, Utils, API, FormHandler, ModalHandler, TableHandler };
}

// ========================================
// CAMERA CONTROLS
// ========================================
class CameraControls {
    constructor() {
        this.isEnabled = true;
        this.isCapturing = false;
        this.statusCheckInterval = null;
        
        this.init();
    }
    
    init() {
        // Bind event listeners
        const toggleBtn = document.getElementById('toggle-camera');
        
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleCamera());
        }
        
        // Start status checking
        this.startStatusChecking();
    }
    
    async toggleCamera() {
        try {
            const response = await fetch('/api/camera/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enabled: !this.isEnabled })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.isEnabled = data.enabled;
                this.updateUI();
            } else {
                this.showError('Lỗi: ' + data.error);
            }
        } catch (error) {
            console.error('Error toggling camera:', error);
            this.showError('Lỗi kết nối: ' + error.message);
        }
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/api/camera/status');
            const data = await response.json();
            
            this.isEnabled = data.enabled;
            this.updateUI();
        } catch (error) {
            console.error('Error checking camera status:', error);
            this.updateStatusIndicator('checking', 'Lỗi kết nối');
        }
    }
    
    startStatusChecking() {
        // Check immediately
        this.checkStatus();
        this.checkAttendanceNotifications();
        
        // Then check every 5 seconds
        this.statusCheckInterval = setInterval(() => {
            this.checkStatus();
            this.checkAttendanceNotifications();
        }, 5000);
    }
    
    async checkAttendanceNotifications() {
        try {
            const response = await fetch('/api/attendance/notifications');
            const data = await response.json();
            
            if (data.notifications && data.notifications.length > 0) {
                data.notifications.forEach(notification => {
                    this.showNotification(notification.message, notification.type);
                });
            }
        } catch (error) {
            console.error('Error checking attendance notifications:', error);
        }
    }
    
    updateUI() {
        const toggleBtn = document.getElementById('toggle-camera');
        
        if (toggleBtn) {
            if (this.isEnabled) {
                toggleBtn.innerHTML = '<i class="fas fa-video-slash me-1"></i>Tắt Camera';
                toggleBtn.className = 'btn btn-danger w-100';
            } else {
                toggleBtn.innerHTML = '<i class="fas fa-video me-1"></i>Bật Camera';
                toggleBtn.className = 'btn btn-warning w-100';
            }
        }
        
        this.updateStatusIndicator(
            this.isEnabled ? 'active' : 'inactive',
            this.isEnabled ? 'Camera đang hoạt động' : 'Camera đã tắt'
        );
    }
    
    updateStatusIndicator(status, text) {
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.getElementById('camera-status-text');
        
        if (indicator) {
            indicator.className = `status-indicator ${status}`;
        }
        
        if (statusText) {
            statusText.textContent = text;
        }
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showError(message) {
        this.showNotification(message, 'danger');
    }
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
}

// Initialize camera controls when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new CameraControls();
});
