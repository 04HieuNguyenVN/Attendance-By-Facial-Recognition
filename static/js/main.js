/**
 * Hệ thống điểm danh - JavaScript chính
 * Quản lý các chức năng chung của ứng dụng
 */

// ========================================
// DOMContentLoaded
// ========================================
document.addEventListener("DOMContentLoaded", () => {
  // Initialize all handlers
  const App = new MainApp();
  App.init();
});

// ========================================
// ATTENDANCE API CLIENT
// ========================================
class AttendanceAPI {
  constructor(fetcher) {
    this.fetcher = fetcher;
  }

  withCacheBuster(path, enable = true) {
    if (!enable) {
      return path;
    }
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}t=${Date.now()}`;
  }

  getSession(options = {}) {
    return this.fetcher(
      this.withCacheBuster(
        "/api/attendance/session",
        options.cacheBust !== false
      ),
      { headers: { "Cache-Control": "no-cache" } }
    );
  }

  openSession(creditClassId, payload = {}) {
    return this.fetcher("/api/attendance/session/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ credit_class_id: creditClassId, ...payload }),
    });
  }

  closeSession(payload = {}) {
    return this.fetcher("/api/attendance/session/close", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  manualMark(sessionId, payload) {
    return this.fetcher(`/api/attendance/session/${sessionId}/mark`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  getTodayAttendance(options = {}) {
    return this.fetcher(
      this.withCacheBuster(
        "/api/attendance/today",
        options.cacheBust !== false
      ),
      { headers: { "Cache-Control": "no-cache" } }
    );
  }

  getStudentHistory(studentId, options = {}) {
    if (!studentId) {
      throw new Error("studentId is required");
    }
    const params = new URLSearchParams();
    if (options.limit) {
      params.set("limit", options.limit);
    }
    if (options.cacheBust !== false) {
      params.set("t", Date.now());
    }
    const query = params.toString();
    return this.fetcher(
      `/api/attendance/history/${encodeURIComponent(studentId)}${
        query ? `?${query}` : ""
      }`,
      { headers: { "Cache-Control": "no-cache" } }
    );
  }

  getStatistics(options = {}) {
    return this.fetcher(
      this.withCacheBuster("/api/statistics", options.cacheBust !== false),
      {
        headers: { "Cache-Control": "no-cache" },
      }
    );
  }

  getActivePresence(options = {}) {
    return this.fetcher(
      this.withCacheBuster("/api/presence/active", options.cacheBust !== false),
      {
        headers: { "Cache-Control": "no-cache" },
      }
    );
  }
}

// ========================================
// MAIN APP CLASS
// ========================================
class MainApp {
  constructor() {
    this.sse = null;
    this.webcamStream = null;
    this.capturedImages = [];
    this.requiredFaceSamples = 3;
    this.maxCapturedImages = 12;
    this.refreshIntervalId = null;
    this.detailModal = null;
    this.detailModalEl = null;
    this.detailElements = {};
    this.classOptions = [];
    this.classOptionsPromise = null;
    this.creditClassOptions = [];
    this.creditClassPromise = null;
    this.sessionElements = {};
    this.activeSession = null;
    this.sessionCountdownInterval = null;
    this.sessionPollInterval = null;
    this.teacherPortalEl = null;
    this.teacherClassList = null;
    this.teacherClassLoading = null;
    this.teacherClassEmpty = null;
    this.teacherClassCount = null;
    this.teacherClassesIndex = {};
    this.teacherClassModal = null;
    this.studentPortalEl = null;
    this.studentClassList = null;
    this.studentSummaryTotalEl = null;
    this.studentSummaryActiveEl = null;
    this.studentClassLoading = null;
    this.studentClassEmpty = null;
    this.studentHistoryList = null;
    this.studentHistoryLoading = null;
    this.studentHistoryEmpty = null;
    this.studentClassSelectEl = null;
    this.studentActionInputs = [];
    this.studentStartBtn = null;
    this.studentStopBtn = null;
    this.studentSessionAlert = null;
    this.studentSessionHint = null;
    this.studentCameraSlot = null;
    this.studentCameraWrapper = null;
    this.studentVideoEl = null;
    this.studentSelectedClassId = "";
    this.studentCurrentAction = "checkin";
    this.studentSessionState = null;
    this.studentCameraActive = false;
    this.studentProfileForm = null;
    this.studentProfileAlert = null;
    this.studentProfileSaveBtn = null;
    this.studentPendingSessionClassId = null;
    this.credentialModal = null;
    this.credentialModalEl = null;
    this.credentialCopyButtons = [];
    this.credentialModalCloseCallback = null;
    this.api = new AttendanceAPI((url, options) =>
      this.fetchJson(url, options)
    );
  }

  init() {
    this.initUI();
    this.initCredentialModal();
    this.initSSE();
    this.initCamera();
    this.initQuickRegister();
    this.initQuickActions();
    this.initDetailModal();
    this.bindDetailHandlers();
    this.initAttendanceSessionPanel();
    this.initTeacherPortal();
    this.initStudentPortal();
    this.startDataRefresh();
    console.log("Hệ thống điểm danh đã khởi tạo thành công");
  }

  initUI() {
    // Add any general UI initializations here
  }

  initCredentialModal() {
    this.credentialModalEl = document.getElementById("credentialModal");
    if (!this.credentialModalEl) {
      window.showCredentialModal = (payload = {}) => {
        if (payload.username || payload.password) {
          alert(
            `Tài khoản: ${payload.username || ""}\nMật khẩu: ${
              payload.password || ""
            }`
          );
        }
      };
      return;
    }

    this.credentialModal = new bootstrap.Modal(this.credentialModalEl);
    this.credentialCopyButtons = Array.from(
      this.credentialModalEl.querySelectorAll("[data-credential-copy]") || []
    );
    this.credentialCopyButtons.forEach((btn) => {
      btn.dataset.defaultLabel = btn.innerHTML;
      btn.addEventListener("click", () => this.handleCredentialCopy(btn));
    });

    this.credentialModalEl.addEventListener("hidden.bs.modal", () => {
      if (typeof this.credentialModalCloseCallback === "function") {
        const callback = this.credentialModalCloseCallback;
        this.credentialModalCloseCallback = null;
        callback();
      }
    });

    window.showCredentialModal = (payload = {}, options = {}) =>
      this.presentCredentialModal(payload, options);
  }

  resetCredentialCopyLabels() {
    if (!this.credentialCopyButtons || !this.credentialCopyButtons.length) {
      return;
    }
    this.credentialCopyButtons.forEach((btn) => {
      if (btn.dataset && btn.dataset.defaultLabel) {
        btn.innerHTML = btn.dataset.defaultLabel;
      }
    });
  }

  async handleCredentialCopy(button) {
    if (!button || !this.credentialModalEl) {
      return;
    }
    const target = button.getAttribute("data-credential-copy");
    if (!target) {
      return;
    }
    const input = this.credentialModalEl.querySelector(
      `[data-credential-input="${target}"]`
    );
    if (!input || !input.value) {
      return;
    }

    const fallback = () => {
      window.prompt("Nhấn Ctrl+C để sao chép", input.value);
    };

    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(input.value);
      } else {
        fallback();
      }
      button.innerHTML = '<i class="fas fa-check me-1"></i>Đã chép';
      setTimeout(() => {
        if (button.dataset && button.dataset.defaultLabel) {
          button.innerHTML = button.dataset.defaultLabel;
        }
      }, 2000);
    } catch (error) {
      console.warn("Không thể sao chép tự động", error);
      fallback();
    }
  }

  presentCredentialModal(payload = {}, options = {}) {
    if (!this.credentialModal || !this.credentialModalEl) {
      return;
    }

    const usernameInput = this.credentialModalEl.querySelector(
      '[data-credential-input="username"]'
    );
    const passwordInput = this.credentialModalEl.querySelector(
      '[data-credential-input="password"]'
    );
    const studentLabel = this.credentialModalEl.querySelector(
      '[data-credential-display="student"]'
    );

    if (usernameInput) {
      usernameInput.value = payload.username || "";
    }
    if (passwordInput) {
      passwordInput.value = payload.password || "";
    }
    if (studentLabel) {
      const studentText =
        payload.full_name && payload.student_id
          ? `${payload.full_name} (${payload.student_id})`
          : payload.full_name || payload.student_id || "-";
      studentLabel.textContent = studentText;
    }

    this.resetCredentialCopyLabels();
    this.credentialModalCloseCallback =
      typeof options.onClose === "function" ? options.onClose : null;
    this.credentialModal.show();
  }

  // ========================================
  // SERVER-SENT EVENTS (SSE)
  // ========================================
  initSSE() {
    this.sse = new EventSource("/api/events/stream");

    this.sse.onopen = () => console.log("SSE connection established");

    this.sse.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (
          !data ||
          !data.type ||
          data.type === "heartbeat" ||
          data.type === "connected"
        ) {
          return;
        }
        if (data.type === "session_updated") {
          this.handleSessionEvent(data.data || null);
          return;
        }
        if (["attendance_marked", "attendance_checkout"].includes(data.type)) {
          this.handleRealtimeEvent(data);
        }
      } catch (e) {
        console.error("Error parsing SSE data:", e);
      }
    };

    this.sse.onerror = (error) => {
      console.error("SSE error:", error);
      this.sse.close();
      setTimeout(() => this.initSSE(), 5000); // Reconnect after 5s
    };

    window.addEventListener("beforeunload", () => {
      if (this.sse) {
        this.sse.close();
      }
    });
  }

  handleRealtimeEvent(eventPacket) {
    const payload = eventPacket.data || {};
    this.showAttendanceNotification(payload, eventPacket.type);
    this.refreshAttendanceList(true);
    this.updateStatistics();
    this.updateActivePresence();

    if (
      this.studentPortalEl &&
      this.studentCameraActive &&
      payload.student_id &&
      this.studentPortalEl.dataset.studentId &&
      String(payload.student_id) === this.studentPortalEl.dataset.studentId
    ) {
      const message =
        eventPacket.type === "attendance_checkout"
          ? "Bạn đã checkout thành công. Camera vẫn đang hoạt động để bạn có thể tiếp tục phiên."
          : "Bạn đã điểm danh thành công. Camera sẽ tiếp tục chạy để bạn có thể checkout hoặc thực hiện thao tác khác.";
      this.showStudentSessionAlert(message, false, false, true);
      this.setStudentSessionHint(
        "Giữ khuôn mặt trong khung hoặc nhấn Tắt camera nếu bạn muốn kết thúc."
      );
    }
  }

  showAttendanceNotification(data = {}, eventType = "attendance_marked") {
    if (!data.student_name) {
      return;
    }
    const isCheckout = eventType === "attendance_checkout";
    const toastClass = isCheckout ? "bg-secondary" : "bg-success";
    const icon = isCheckout ? "fa-door-closed" : "fa-check-circle";
    const actionText = isCheckout ? "đã checkout!" : "đã điểm danh!";
    const notificationHtml = `
      <div class="toast align-items-center text-white ${toastClass} border-0 show attendance-toast" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="d-flex">
          <div class="toast-body">
            <i class="fas ${icon} me-2"></i>
            <strong>${data.student_name}</strong> ${actionText}
          </div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
      </div>`;
    const toastDiv = document.createElement("div");
    toastDiv.innerHTML = notificationHtml;
    document.body.appendChild(toastDiv);
    setTimeout(() => toastDiv.remove(), 5000);
  }

  // ========================================
  // CAMERA
  // ========================================
  initCamera() {
    const toggleSwitch = document.getElementById("toggle-camera");
    const switchLabel = document.getElementById("camera-switch-label");

    if (!toggleSwitch || !switchLabel) {
      return;
    }

    const updateStatus = async () => {
      try {
        const response = await fetch("/api/camera/status");
        const data = await response.json();
        toggleSwitch.checked = data.enabled;
        switchLabel.textContent = data.enabled ? "Bật" : "Tắt";
      } catch (error) {
        console.error("Error checking camera status:", error);
      }
    };

    toggleSwitch.addEventListener("change", async function () {
      const isEnabled = this.checked;
      switchLabel.textContent = isEnabled ? "Đang bật..." : "Đang tắt...";
      try {
        const response = await fetch("/api/camera/toggle", { method: "POST" });
        const data = await response.json();
        if (data.success) {
          switchLabel.textContent = data.enabled ? "Bật" : "Tắt";
          if (data.enabled) {
            setTimeout(() => location.reload(), 500);
          }
        }
      } catch (error) {
        console.error("Error toggling camera:", error);
      }
    });

    updateStatus();
    setInterval(updateStatus, 10000);
  }

  // ========================================
  // QUICK REGISTER
  // ========================================
  initQuickRegister() {
    const modalEl = document.getElementById("quickRegisterModal");
    if (!modalEl) return;

    const quickRegisterModal = new bootstrap.Modal(modalEl);
    const quickRegisterBtn = document.getElementById("quick-register-btn");
    const startWebcamBtn = document.getElementById("start-webcam-btn");
    const stopWebcamBtn = document.getElementById("stop-webcam-btn");
    const captureBtn = document.getElementById("capture-btn");
    const retakeBtn = document.getElementById("retake-btn");
    const submitBtn = document.getElementById("submit-register-btn");
    const fileInput = document.getElementById("reg-face-images");
    const capturedList = document.getElementById("captured-images-list");

    if (
      !quickRegisterBtn ||
      !startWebcamBtn ||
      !captureBtn ||
      !retakeBtn ||
      !submitBtn ||
      !fileInput ||
      !capturedList
    ) {
      return;
    }

    quickRegisterBtn.addEventListener("click", () => quickRegisterModal.show());
    modalEl.addEventListener("show.bs.modal", () => {
      this.populateQuickRegisterClassOptions();
      this.updateCapturedImagesUI();
      this.toggleRegisterCameraUI(false);
    });
    this.populateQuickRegisterClassOptions();

    startWebcamBtn.addEventListener("click", async () => {
      try {
        this.webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 320, height: 240 },
        });
        this.toggleRegisterCameraUI(true);
      } catch (error) {
        alert("Không thể truy cập camera. Lỗi: " + error.message);
      }
    });

    if (stopWebcamBtn) {
      stopWebcamBtn.addEventListener("click", () => this.stopWebcam());
    }

    captureBtn.addEventListener("click", () => {
      if (!this.webcamStream) {
        alert("Vui lòng bật camera trước khi chụp.");
        return;
      }
      if (this.capturedImages.length >= this.maxCapturedImages) {
        alert(
          `Bạn chỉ có thể lưu tối đa ${this.maxCapturedImages} ảnh chụp trực tiếp mỗi lần.`
        );
        return;
      }
      const canvas = document.getElementById("capture-canvas");
      const video = document.getElementById("webcam-preview");
      const context = canvas.getContext("2d");
      canvas.width = 320;
      canvas.height = 240;
      context.drawImage(video, 0, 0, 320, 240);
      this.capturedImages.push(canvas.toDataURL("image/jpeg"));
      this.updateCapturedImagesUI();
    });

    retakeBtn.addEventListener("click", () => this.clearCapturedImages());
    fileInput.addEventListener("change", () => this.updateCapturedImagesUI());
    capturedList.addEventListener("click", (event) => {
      const removeBtn = event.target.closest(".remove-capture");
      if (!removeBtn) {
        return;
      }
      const index = Number(removeBtn.dataset.index);
      if (!Number.isNaN(index)) {
        this.capturedImages.splice(index, 1);
        this.updateCapturedImagesUI();
      }
    });

    submitBtn.addEventListener("click", async () => {
      const form = document.getElementById("quickRegisterForm");
      const formData = new FormData(form);
      const resultDiv = document.getElementById("register-result");
      const totalSamples = this.getTotalPreparedSamples();

      if (totalSamples < this.requiredFaceSamples) {
        if (resultDiv) {
          resultDiv.innerHTML = `<div class="alert alert-warning">Cần tối thiểu ${this.requiredFaceSamples} ảnh khuôn mặt (đã có ${totalSamples}).</div>`;
        }
        return;
      }

      this.capturedImages.forEach((imageData, idx) => {
        formData.append(`image_data_${idx}`, imageData);
      });

      submitBtn.disabled = true;
      submitBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin me-1"></i>Đang đăng ký...';

      try {
        const response = await fetch("/api/quick-register", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (response.ok && data.success) {
          if (resultDiv) {
            resultDiv.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
          }

          const resetFormState = () => {
            this.resetQuickRegisterState();
            if (resultDiv) {
              resultDiv.innerHTML = "";
            }
          };

          const finishAndReload = () => {
            resetFormState();
            quickRegisterModal.hide();
            window.location.reload();
          };

          if (data.credentials) {
            quickRegisterModal.hide();
            const credentialPayload = {
              student_id:
                data.credentials.student_id || formData.get("student_id") || "",
              full_name:
                data.credentials.full_name || formData.get("full_name") || "",
              username: data.credentials.username,
              password: data.credentials.password,
            };
            this.presentCredentialModal(credentialPayload, {
              onClose: finishAndReload,
            });
          } else {
            setTimeout(finishAndReload, 1500);
          }
        } else if (resultDiv) {
          resultDiv.innerHTML = `<div class="alert alert-danger">${
            data.error || "Đăng ký thất bại"
          }</div>`;
        }
      } catch (error) {
        if (resultDiv) {
          resultDiv.innerHTML = `<div class="alert alert-danger">Lỗi: ${error.message}</div>`;
        }
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-save me-1"></i>Đăng ký';
      }
    });

    modalEl.addEventListener("hidden.bs.modal", () => {
      this.stopWebcam();
      this.resetQuickRegisterState();
    });
  }

  async loadClassOptions(forceReload = false) {
    if (!forceReload && this.classOptions.length) {
      return this.classOptions;
    }
    if (!forceReload && this.classOptionsPromise) {
      return this.classOptionsPromise;
    }

    const loader = (async () => {
      try {
        const response = await fetch(`/api/classes?t=${Date.now()}`);
        if (!response.ok) {
          throw new Error("Không thể tải danh sách lớp học");
        }
        const payload = await response.json();
        this.classOptions = payload.success ? payload.data || [] : [];
      } catch (error) {
        console.error("Error loading class options:", error);
        this.classOptions = [];
      } finally {
        this.classOptionsPromise = null;
      }
      return this.classOptions;
    })();

    if (!forceReload) {
      this.classOptionsPromise = loader;
    }

    return loader;
  }

  async populateQuickRegisterClassOptions(forceReload = false) {
    const datalist = document.getElementById("quick-register-class-options");
    if (!datalist) {
      return;
    }

    const classes = await this.loadClassOptions(forceReload);
    datalist.innerHTML = "";

    const seenNames = new Set();
    classes.forEach((cls) => {
      const name = cls.class_name || cls.class_code;
      if (!name || seenNames.has(name)) {
        return;
      }
      seenNames.add(name);
      const option = document.createElement("option");
      option.value = name;
      if (cls.class_code && cls.class_code !== name) {
        option.label = `${cls.class_code} · ${name}`;
      }
      datalist.appendChild(option);
    });
  }

  toggleRegisterCameraUI(isStreaming) {
    const videoEl = document.getElementById("webcam-preview");
    const placeholder = document.getElementById("camera-placeholder");
    const startBtn = document.getElementById("start-webcam-btn");
    const captureBtn = document.getElementById("capture-btn");
    const stopBtn = document.getElementById("stop-webcam-btn");

    if (videoEl) {
      videoEl.classList.toggle("d-none", !isStreaming);
      if (isStreaming && this.webcamStream) {
        videoEl.srcObject = this.webcamStream;
      } else if (!isStreaming) {
        videoEl.srcObject = null;
      }
    }
    if (placeholder) {
      placeholder.style.display = isStreaming ? "none" : "flex";
    }
    if (startBtn) {
      startBtn.classList.toggle("d-none", isStreaming);
    }
    if (captureBtn) {
      captureBtn.classList.toggle("d-none", !isStreaming);
      captureBtn.disabled = !isStreaming;
    }
    if (stopBtn) {
      stopBtn.classList.toggle("d-none", !isStreaming);
    }
  }

  stopWebcam() {
    if (this.webcamStream) {
      this.webcamStream.getTracks().forEach((track) => track.stop());
      this.webcamStream = null;
    }
    this.toggleRegisterCameraUI(false);
  }

  getUploadedFileCount() {
    const fileInput = document.getElementById("reg-face-images");
    return fileInput && fileInput.files ? fileInput.files.length : 0;
  }

  getTotalPreparedSamples() {
    return this.capturedImages.length + this.getUploadedFileCount();
  }

  clearCapturedImages() {
    this.capturedImages = [];
    this.updateCapturedImagesUI();
  }

  resetQuickRegisterState() {
    const form = document.getElementById("quickRegisterForm");
    if (form) {
      form.reset();
    }
    this.clearCapturedImages();
    const fileInput = document.getElementById("reg-face-images");
    if (fileInput) {
      fileInput.value = "";
    }
    const resultDiv = document.getElementById("register-result");
    if (resultDiv) {
      resultDiv.innerHTML = "";
    }
    this.toggleRegisterCameraUI(false);
  }

  updateCapturedImagesUI() {
    const listEl = document.getElementById("captured-images-list");
    const counterEl = document.getElementById("capture-counter");
    const retakeBtn = document.getElementById("retake-btn");
    const totalSamples = this.getTotalPreparedSamples();

    if (listEl) {
      listEl.innerHTML = "";
      this.capturedImages.forEach((imageData, idx) => {
        const col = document.createElement("div");
        col.className = "col-4";
        col.innerHTML = `
          <div class="captured-thumb border">
            <img src="${imageData}" class="img-fluid" alt="Ảnh ${idx + 1}">
            <button type="button" class="btn btn-sm btn-danger remove-capture" data-index="${idx}" aria-label="Xóa ảnh ${
          idx + 1
        }">
              <i class="fas fa-times"></i>
            </button>
          </div>
        `;
        listEl.appendChild(col);
      });
    }

    if (counterEl) {
      if (totalSamples > 0) {
        counterEl.textContent = `Đã chuẩn bị ${totalSamples}/${
          this.requiredFaceSamples
        } ảnh (camera: ${
          this.capturedImages.length
        }, file: ${this.getUploadedFileCount()}).`;
      } else {
        counterEl.textContent = `Cần tối thiểu ${this.requiredFaceSamples} ảnh khuôn mặt rõ nét.`;
      }
      counterEl.classList.toggle(
        "text-danger",
        totalSamples < this.requiredFaceSamples
      );
    }

    if (retakeBtn) {
      retakeBtn.classList.toggle("d-none", this.capturedImages.length === 0);
    }
  }

  // ========================================
  // QUICK ACTIONS
  // ========================================
  initQuickActions() {
    const reloadFacesBtn = document.getElementById("reload-faces");
    if (reloadFacesBtn) {
      reloadFacesBtn.addEventListener("click", async (e) => {
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.innerHTML =
          '<i class="fas fa-spinner fa-spin me-2"></i>Đang cập nhật...';
        try {
          const res = await fetch("/update_faces", { method: "POST" });
          if (res.ok) {
            alert("Cập nhật dữ liệu khuôn mặt thành công!");
            location.reload();
          } else {
            alert("Lỗi cập nhật dữ liệu.");
          }
        } catch (err) {
          alert("Lỗi: " + err.message);
        } finally {
          btn.disabled = false;
          btn.innerHTML =
            '<i class="fas fa-sync-alt me-2"></i>Cập nhật dữ liệu';
        }
      });
    }

    const refreshAttendanceBtn = document.getElementById("refresh-attendance");
    if (refreshAttendanceBtn) {
      refreshAttendanceBtn.addEventListener("click", () =>
        this.refreshAttendanceList()
      );
    }
  }

  // ========================================
  // DATA REFRESH
  // ========================================
  startDataRefresh() {
    this.refreshAttendanceList();
    this.updateStatistics();
    this.updateActivePresence();

    this.refreshIntervalId = setInterval(() => {
      this.refreshAttendanceList();
      this.updateStatistics();
      this.updateActivePresence();
    }, 5000); // Refresh every 5 seconds
  }

  // ========================================
  // TEACHER PORTAL
  // ========================================
  initTeacherPortal() {
    this.teacherPortalEl = document.getElementById("teacher-portal");
    if (!this.teacherPortalEl) {
      return;
    }

    this.teacherClassList = document.getElementById("teacher-class-list");
    this.teacherClassLoading = document.getElementById("teacher-class-loading");
    this.teacherClassEmpty = document.getElementById("teacher-class-empty");
    this.teacherClassCount = document.getElementById("teacher-class-count");
    this.teacherEmptyDefault = this.teacherClassEmpty
      ? this.teacherClassEmpty.innerHTML
      : "";
    this.teacherStatTotalEl = document.querySelector("#teacher-stat-total h4");
    this.teacherStatActiveEl = document.querySelector(
      "#teacher-stat-active h4"
    );
    this.teacherStatStudentsEl = document.querySelector(
      "#teacher-stat-students h4"
    );
    this.teacherClassModalEl = document.getElementById("teacherClassModal");
    if (this.teacherClassModalEl && typeof bootstrap !== "undefined") {
      this.teacherClassModal = new bootstrap.Modal(this.teacherClassModalEl);
    }

    const refreshBtn = document.getElementById("teacher-refresh");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", () =>
        this.fetchTeacherClasses(true)
      );
    }

    this.teacherPortalEl.addEventListener("click", (event) => {
      const actionBtn = event.target.closest(
        "[data-action='teacher-view-class']"
      );
      if (actionBtn) {
        const classId = Number(actionBtn.getAttribute("data-class-id"));
        if (classId) this.openTeacherClassModal(classId);
        return;
      }

      // Handle manual mark actions
      const markCheckin = event.target.closest(
        "[data-action='teacher-mark-checkin']"
      );
      if (markCheckin) {
        const studentId = markCheckin.getAttribute("data-student-id");
        const sessionId = markCheckin.getAttribute("data-session-id");
        if (studentId && sessionId) {
          this.teacherManualMark(sessionId, studentId, "checkin");
        }
        return;
      }

      const markCheckout = event.target.closest(
        "[data-action='teacher-mark-checkout']"
      );
      if (markCheckout) {
        const studentId = markCheckout.getAttribute("data-student-id");
        const sessionId = markCheckout.getAttribute("data-session-id");
        if (studentId && sessionId) {
          this.teacherManualMark(sessionId, studentId, "checkout");
        }
        return;
      }
    });

    this.fetchTeacherClasses();
  }

  async fetchTeacherClasses(forceReload = false) {
    if (!this.teacherPortalEl) {
      return;
    }
    this.toggleTeacherLoading(true);
    const params = new URLSearchParams();
    const teacherId = this.teacherPortalEl.dataset.teacherId;
    if (teacherId) {
      params.set("teacher_id", teacherId);
    }
    if (forceReload) {
      params.set("t", Date.now());
    }
    const query = params.toString();
    const endpoint = query
      ? `/api/teacher/credit-classes?${query}`
      : "/api/teacher/credit-classes";
    try {
      const data = await this.fetchJson(endpoint);
      const classes = data.data || [];
      this.teacherClassesIndex = {};
      classes.forEach((cls) => {
        if (cls && cls.id) {
          this.teacherClassesIndex[cls.id] = cls;
        }
      });
      this.renderTeacherClasses(classes);
      this.updateTeacherStats(classes);
    } catch (error) {
      this.renderTeacherClasses([]);
      this.showTeacherEmptyState(
        error.message || "Không thể tải dữ liệu",
        true
      );
    } finally {
      this.toggleTeacherLoading(false);
    }
  }

  renderTeacherClasses(classes = []) {
    if (!this.teacherClassList) {
      return;
    }
    this.teacherClassList.innerHTML = "";
    if (!classes.length) {
      this.showTeacherEmptyState(this.teacherEmptyDefault, false);
      if (this.teacherClassCount) {
        this.teacherClassCount.textContent = "0 lớp";
      }
      return;
    }
    this.showTeacherEmptyState("", false, true);
    if (this.teacherClassCount) {
      this.teacherClassCount.textContent = `${classes.length} lớp`;
    }

    const fragment = document.createDocumentFragment();
    classes.forEach((cls) => {
      const col = document.createElement("div");
      col.className = "col-12";
      const studentCount = Number(cls.student_count || 0);
      const schedule = [cls.semester, cls.academic_year]
        .filter(Boolean)
        .join(" · ");
      const scheduleLabel = schedule || "Lịch chưa cập nhật";
      const activeSession = cls.active_session;
      const sessionHtml = activeSession
        ? `<div class="alert alert-success py-2 mb-3"><i class="fas fa-fingerprint me-2"></i>Phiên đang mở đến ${this.formatDateTime(
            activeSession.expires_at ||
              activeSession.checkout_deadline ||
              activeSession.checkin_deadline
          )}</div>`
        : `<div class="alert alert-secondary py-2 mb-3"><i class="fas fa-clock me-2"></i>Chưa có phiên điểm danh nào mở</div>`;
      col.innerHTML = `
        <div class="teacher-class-card h-100">
          <div class="d-flex justify-content-between align-items-start mb-2">
            <div>
              <h6 class="mb-1">${this.escapeHtml(
                cls.display_name || cls.subject_name || "Lớp tín chỉ"
              )}</h6>
              <div class="teacher-class-meta">${this.escapeHtml(
                scheduleLabel
              )}</div>
            </div>
            <span class="badge bg-light text-dark">${studentCount} SV</span>
          </div>
          <p class="text-muted mb-2">
            <i class="fas fa-chalkboard me-1"></i>
            ${this.escapeHtml(cls.room || "Chưa có phòng học")}
          </p>
          ${sessionHtml}
          <div class="d-flex gap-2 flex-wrap">
            <button class="btn btn-sm btn-outline-primary" data-action="teacher-view-class" data-class-id="${
              cls.id
            }">
              <i class="fas fa-layer-group me-1"></i>Chi tiết lớp
            </button>
          </div>
        </div>`;
      fragment.appendChild(col);
    });
    this.teacherClassList.appendChild(fragment);
  }

  showTeacherEmptyState(message = "", isError = false, hide = false) {
    if (!this.teacherClassEmpty) {
      return;
    }
    if (hide) {
      this.teacherClassEmpty.classList.add("d-none");
      return;
    }
    this.teacherClassEmpty.classList.remove("d-none");
    this.teacherClassEmpty.innerHTML = message
      ? `<p class="${
          isError ? "text-danger" : "text-muted"
        } mb-0">${message}</p>`
      : this.teacherEmptyDefault;
  }

  toggleTeacherLoading(isLoading) {
    if (this.teacherClassLoading) {
      this.teacherClassLoading.classList.toggle("d-none", !isLoading);
    }
    if (this.teacherClassList) {
      this.teacherClassList.classList.toggle("d-none", isLoading);
    }
  }

  updateTeacherStats(classes = []) {
    const total = classes.length;
    const active = classes.filter((cls) => cls.active_session).length;
    const totalStudents = classes.reduce(
      (sum, cls) => sum + Number(cls.student_count || 0),
      0
    );
    if (this.teacherStatTotalEl) {
      this.teacherStatTotalEl.textContent = total;
    }
    if (this.teacherStatActiveEl) {
      this.teacherStatActiveEl.textContent = active;
    }
    if (this.teacherStatStudentsEl) {
      this.teacherStatStudentsEl.textContent = totalStudents;
    }
  }

  async openTeacherClassModal(classId) {
    const classInfo = this.teacherClassesIndex[classId];
    if (!classInfo || !this.teacherClassModal) {
      return;
    }
    const titleEl = document.getElementById("teacher-class-modal-title");
    const metaEl = document.getElementById("teacher-class-modal-meta");
    const studentsEl = document.getElementById("teacher-class-students");
    const sessionsEl = document.getElementById("teacher-class-sessions");
    if (titleEl) {
      titleEl.textContent =
        classInfo.display_name || classInfo.subject_name || "Lớp tín chỉ";
    }
    if (metaEl) {
      const metaPieces = [
        classInfo.semester,
        classInfo.academic_year,
        classInfo.room,
      ]
        .filter(Boolean)
        .join(" · ");
      metaEl.textContent = metaPieces || "Chưa có thông tin chi tiết";
    }
    if (studentsEl) {
      studentsEl.innerHTML =
        '<div class="text-center py-3 text-muted"><i class="fas fa-spinner fa-spin me-2"></i>Đang tải danh sách sinh viên...</div>';
    }
    if (sessionsEl) {
      sessionsEl.innerHTML =
        '<div class="text-center py-3 text-muted"><i class="fas fa-spinner fa-spin me-2"></i>Đang tải phiên điểm danh...</div>';
    }
    this.teacherClassModal.show();

    // store meta on modal element for use by students rendering
    try {
      if (this.teacherClassModalEl) {
        this.teacherClassModalEl.dataset.creditClassId = classId;
      }
    } catch (e) {}

    try {
      const [roster, sessions] = await Promise.all([
        this.fetchJson(`/api/teacher/credit-classes/${classId}/students`),
        this.fetchJson(`/api/teacher/credit-classes/${classId}/sessions`),
      ]);
      // determine active session id if any
      let activeSessionId = null;
      if (sessions && sessions.sessions && sessions.sessions.length) {
        for (const s of sessions.sessions) {
          if ((s.status || "").toLowerCase() === "open") {
            activeSessionId = s.id;
            break;
          }
        }
      }
      try {
        if (this.teacherClassModalEl) {
          this.teacherClassModalEl.dataset.sessionId = activeSessionId || "";
        }
      } catch (e) {}

      this.renderTeacherModalStudents((roster && roster.students) || []);
      this.renderTeacherModalSessions((sessions && sessions.sessions) || []);
    } catch (error) {
      if (studentsEl) {
        studentsEl.innerHTML = `<p class="text-danger mb-0">${error.message}</p>`;
      }
      if (sessionsEl) {
        sessionsEl.innerHTML = `<p class="text-danger mb-0">${error.message}</p>`;
      }
    }
  }

  renderTeacherModalStudents(students = []) {
    const container = document.getElementById("teacher-class-students");
    if (!container) {
      return;
    }
    if (!students.length) {
      container.innerHTML =
        '<p class="text-muted mb-0">Chưa có sinh viên nào trong lớp này.</p>';
      return;
    }
    const sessionId =
      (this.teacherClassModalEl &&
        this.teacherClassModalEl.dataset.sessionId) ||
      "";
    const rows = students
      .map((student) => {
        const isPresent = !!student.is_present_today;
        const checkedOut = !!student.checked_out;
        const studentId = this.escapeHtml(student.student_id || "");
        const displayName = this.escapeHtml(
          student.full_name || student.student_id || ""
        );
        const className = this.escapeHtml(student.class_name || "N/A");

        const actionHtml = isPresent
          ? checkedOut
            ? `<span class="badge bg-secondary">Đã checkout</span>`
            : `<button class="btn btn-sm btn-outline-secondary" data-action="teacher-mark-checkout" data-student-id="${studentId}" data-session-id="${sessionId}"><i class="fas fa-door-closed me-1"></i>Checkout</button>`
          : `<button class="btn btn-sm btn-primary" data-action="teacher-mark-checkin" data-student-id="${studentId}" data-session-id="${sessionId}"><i class="fas fa-user-check me-1"></i>Điểm danh</button>`;

        return `
          <div class="d-flex justify-content-between border-bottom py-2 align-items-center">
            <div>
              <div class="fw-semibold">${displayName}</div>
              <small class="text-muted">MSSV: ${studentId}</small>
            </div>
            <div class="text-end">
              <div class="mb-1"><small class="text-muted">${className}</small></div>
              ${actionHtml}
            </div>
          </div>`;
      })
      .join("");
    container.innerHTML = rows;
  }

  async teacherManualMark(sessionId, studentId, action) {
    if (!confirm(`Bạn có chắc muốn thực hiện ${action} cho MSSV ${studentId}?`))
      return;
    try {
      const payload = { student_id: studentId, action };
      const res = await this.api.manualMark(sessionId, payload);
      if (res && res.success) {
        // reload modal roster to reflect status
        const classId =
          this.teacherClassModalEl &&
          this.teacherClassModalEl.dataset.creditClassId;
        if (classId) {
          this.openTeacherClassModal(Number(classId));
        }
      } else {
        alert(res && (res.message || "Thao tác thất bại"));
      }
    } catch (err) {
      alert("Lỗi khi gọi API: " + (err.message || err));
    }
  }

  renderTeacherModalSessions(sessions = []) {
    const container = document.getElementById("teacher-class-sessions");
    if (!container) {
      return;
    }
    if (!sessions.length) {
      container.innerHTML =
        '<p class="text-muted mb-0">Chưa có phiên điểm danh nào được ghi nhận.</p>';
      return;
    }
    const rows = sessions
      .map((session) => {
        const status = (session.status || "").toLowerCase();
        const badgeClass =
          status === "open"
            ? "bg-success"
            : status === "scheduled"
            ? "bg-warning text-dark"
            : "bg-secondary";
        return `
          <div class="d-flex justify-content-between border-bottom py-2">
            <div>
              <div class="fw-semibold">${this.formatDateTime(
                session.opened_at || session.session_date
              )}</div>
              <small class="text-muted">Kết thúc: ${this.formatDateTime(
                session.closed_at ||
                  session.checkout_deadline ||
                  session.checkin_deadline
              )}</small>
            </div>
            <span class="badge ${badgeClass}">${session.status || "--"}</span>
          </div>`;
      })
      .join("");
    container.innerHTML = rows;
  }

  // ========================================
  // STUDENT PORTAL
  // ========================================
  initStudentPortal() {
    this.studentPortalEl = document.getElementById("student-portal");
    if (!this.studentPortalEl) {
      return;
    }
    this.studentClassList = document.getElementById("student-class-list");
    this.studentClassLoading = document.getElementById("student-class-loading");
    this.studentClassEmpty = document.getElementById("student-class-empty");
    this.studentClassEmptyDefault = this.studentClassEmpty
      ? this.studentClassEmpty.innerHTML
      : "";
    this.studentSummaryTotalEl = document.getElementById(
      "student-summary-total"
    );
    this.studentSummaryActiveEl = document.getElementById(
      "student-summary-active"
    );
    this.studentHistoryList = document.getElementById("student-history-list");
    this.studentHistoryLoading = document.getElementById(
      "student-history-loading"
    );
    this.studentHistoryEmpty = document.getElementById("student-history-empty");
    this.studentHistoryEmptyDefault = this.studentHistoryEmpty
      ? this.studentHistoryEmpty.innerHTML
      : "";
    this.studentClassSelectEl = document.getElementById(
      "student-credit-class-select"
    );
    this.studentActionInputs = Array.from(
      document.querySelectorAll('input[name="student-action"]')
    );
    this.studentStartBtn = document.getElementById("student-start-camera");
    this.studentStopBtn = document.getElementById("student-stop-camera");
    this.studentSessionAlert = document.getElementById("student-session-alert");
    this.studentSessionHint = document.getElementById("student-session-hint");
    this.studentCameraSlot = document.getElementById("student-camera-slot");
    this.studentCameraWrapper = document.getElementById(
      "student-camera-wrapper"
    );
    this.studentVideoEl = document.getElementById("video-stream");
    this.studentProfileForm = document.getElementById("student-profile-form");
    this.studentProfileAlert = document.getElementById("student-profile-alert");
    this.studentProfileSaveBtn = document.getElementById(
      "student-profile-save"
    );

    if (this.studentCameraSlot && this.studentCameraWrapper) {
      this.studentCameraSlot.appendChild(this.studentCameraWrapper);
    }

    if (this.studentVideoEl) {
      this.studentVideoEl.removeAttribute("src");
      this.studentVideoEl.classList.add("d-none");
      this.studentVideoEl.addEventListener("load", () => {
        if (this.studentVideoEl.src) {
          this.studentVideoEl.classList.remove("d-none");
          this.showStudentSessionAlert(
            "Camera đang hoạt động. Giữ khuôn mặt trong khung.",
            false
          );
        }
      });
      this.studentVideoEl.addEventListener("error", () => {
        this.stopStudentCamera(
          "Không thể kết nối camera. Vui lòng kiểm tra và thử lại."
        );
      });
    }

    this.bindStudentCameraControls();
    if (this.studentProfileForm) {
      this.studentProfileForm.addEventListener("submit", (event) => {
        event.preventDefault();
        this.submitStudentProfile();
      });
    }

    const refreshBtn = document.getElementById("student-refresh");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", (event) => {
        event.preventDefault();
        this.fetchStudentClasses(true);
      });
    }
    const historyBtn = document.getElementById("student-history-refresh");
    if (historyBtn) {
      historyBtn.addEventListener("click", (event) => {
        event.preventDefault();
        this.fetchStudentHistory(true);
      });
    }

    if (this.studentPortalEl.dataset.studentId) {
      this.fetchStudentClasses();
      this.fetchStudentHistory();
    }
  }

  bindStudentCameraControls() {
    if (!this.studentPortalEl) {
      return;
    }

    if (this.studentActionInputs.length) {
      this.studentActionInputs.forEach((input) => {
        input.addEventListener("change", () => {
          if (!input.checked) {
            return;
          }
          this.studentCurrentAction = input.value || "checkin";
          if (this.studentCameraActive) {
            this.stopStudentCamera(
              "Đã thay đổi hành động. Camera được đặt lại."
            );
          }
        });
      });
    }

    if (this.studentClassSelectEl) {
      this.studentClassSelectEl.addEventListener("change", () => {
        this.studentSelectedClassId = this.studentClassSelectEl.value;
        this.studentSessionState = null;
        if (this.studentStartBtn) {
          this.studentStartBtn.disabled = true;
        }
        if (this.studentCameraActive) {
          this.stopStudentCamera(
            "Đã thay đổi lớp tín chỉ. Camera được đặt lại."
          );
        }
        this.validateStudentSession();
      });
    }

    if (this.studentStartBtn) {
      this.studentStartBtn.addEventListener("click", async (event) => {
        event.preventDefault();
        await this.startStudentCamera();
      });
    }

    if (this.studentStopBtn) {
      this.studentStopBtn.addEventListener("click", (event) => {
        event.preventDefault();
        this.stopStudentCamera();
      });
    }
  }

  setStudentProfileAlert(message = "", variant = "info") {
    if (!this.studentProfileAlert) {
      return;
    }
    const alertEl = this.studentProfileAlert;
    alertEl.classList.remove("alert-info", "alert-success", "alert-danger");
    if (!message) {
      alertEl.classList.add("d-none");
      alertEl.textContent = "";
      return;
    }
    alertEl.textContent = message;
    alertEl.classList.remove("d-none");
    alertEl.classList.add(`alert-${variant}`);
  }

  updateStudentProfileSummary(student = {}) {
    if (!student) {
      return;
    }
    const nameEl = document.getElementById("student-profile-name");
    if (nameEl && student.full_name) {
      nameEl.textContent = student.full_name;
    }
    const codeEl = document.getElementById("student-profile-code");
    if (codeEl && student.student_id) {
      codeEl.textContent = `MSSV: ${student.student_id}`;
    }
    const emailEl = document.getElementById("student-profile-email");
    if (emailEl) {
      emailEl.textContent = student.email || "—";
    }
    const phoneEl = document.getElementById("student-profile-phone");
    if (phoneEl) {
      phoneEl.textContent = student.phone || "—";
    }

    if (this.studentProfileForm) {
      const nameField = document.getElementById(
        "student-profile-field-full-name"
      );
      if (nameField && student.full_name) {
        nameField.value = student.full_name;
      }
      const emailField = document.getElementById("student-profile-field-email");
      if (emailField) {
        emailField.value = student.email || "";
      }
      const phoneField = document.getElementById("student-profile-field-phone");
      if (phoneField) {
        phoneField.value = student.phone || "";
      }
    }

    if (this.studentPortalEl) {
      if (student.student_id) {
        this.studentPortalEl.dataset.studentId = student.student_id;
      }
      this.studentPortalEl.dataset.studentName = student.full_name || "";
      this.studentPortalEl.dataset.studentEmail = student.email || "";
      this.studentPortalEl.dataset.studentPhone = student.phone || "";
    }
  }

  async submitStudentProfile() {
    if (!this.studentProfileForm) {
      return;
    }

    const formData = new FormData(this.studentProfileForm);
    const payload = {
      full_name: (formData.get("full_name") || "").trim(),
      email: (formData.get("email") || "").trim(),
      phone: (formData.get("phone") || "").trim(),
    };

    if (!payload.full_name) {
      this.setStudentProfileAlert("Họ và tên không được để trống.", "danger");
      return;
    }

    this.setStudentProfileAlert("Đang lưu thông tin...", "info");
    if (this.studentProfileSaveBtn) {
      const btn = this.studentProfileSaveBtn;
      if (!btn.dataset.originalText) {
        btn.dataset.originalText = btn.innerHTML;
      }
      btn.disabled = true;
      btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Đang lưu...';
    }

    try {
      const response = await this.fetchJson("/api/student/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      this.setStudentProfileAlert(
        response.message || "Đã lưu thông tin cá nhân",
        "success"
      );
      if (response.student) {
        this.updateStudentProfileSummary(response.student);
      }
    } catch (error) {
      this.setStudentProfileAlert(
        error.message || "Không thể lưu thông tin",
        "danger"
      );
    } finally {
      if (this.studentProfileSaveBtn) {
        const btn = this.studentProfileSaveBtn;
        btn.disabled = false;
        btn.innerHTML =
          btn.dataset.originalText ||
          '<i class="fas fa-save me-1"></i>Lưu thay đổi';
      }
    }
  }

  async fetchStudentClasses(forceReload = false) {
    if (!this.studentPortalEl || !this.studentPortalEl.dataset.studentId) {
      return;
    }
    this.toggleStudentClassLoading(true);
    const params = new URLSearchParams();
    const studentId = this.studentPortalEl.dataset.studentId;
    if (studentId) {
      params.set("student_id", studentId);
    }
    if (forceReload) {
      params.set("t", Date.now());
    }
    const query = params.toString();
    const endpoint = query
      ? `/api/student/credit-classes?${query}`
      : "/api/student/credit-classes";
    try {
      const data = await this.fetchJson(endpoint);
      const classes = data.classes || [];
      this.renderStudentClasses(classes);
      this.populateStudentClassSelect(classes);
      this.updateStudentSummary(data.summary || {});
    } catch (error) {
      this.renderStudentClasses([]);
      this.showStudentClassEmpty(
        error.message || "Không thể tải dữ liệu",
        true
      );
    } finally {
      this.toggleStudentClassLoading(false);
    }
  }

  renderStudentClasses(classes = []) {
    if (!this.studentClassList) {
      return;
    }
    this.studentClassList.innerHTML = "";
    if (!classes.length) {
      this.showStudentClassEmpty("Chưa có lớp tín chỉ nào.");
      return;
    }
    this.showStudentClassEmpty("", false, true);
    const fragment = document.createDocumentFragment();
    classes.forEach((cls) => {
      const col = document.createElement("div");
      col.className = "col-12 col-md-6";
      const activeSession = cls.active_session;
      const isSessionOnly = Boolean(cls.is_session_only);
      const sessionPill = activeSession
        ? `<span class="session-pill bg-success text-white"><i class="fas fa-satellite-dish me-1"></i>Đang mở</span>`
        : `<span class="session-pill bg-light text-muted"><i class="fas fa-clock me-1"></i>Đang đóng</span>`;
      const sessionOnlyBadge = isSessionOnly
        ? '<span class="badge bg-warning text-dark ms-2">Chưa ghi danh</span>'
        : "";
      col.innerHTML = `
        <div class="student-class-card h-100">
          <div class="d-flex justify-content-between align-items-start mb-2">
            <div>
              <h6 class="mb-1">${this.escapeHtml(
                cls.subject_name || cls.display_name || "Lớp tín chỉ"
              )}</h6>
              <div class="class-code">${this.escapeHtml(
                cls.credit_code || "---"
              )}</div>
            </div>
            <div class="d-flex align-items-center">${sessionPill}${sessionOnlyBadge}</div>
          </div>
          <p class="mb-1 text-muted"><i class="fas fa-location-dot me-1"></i>${this.escapeHtml(
            cls.room || "Chưa có phòng"
          )}</p>
          <p class="mb-3 text-muted small">${this.escapeHtml(
            cls.schedule_info || "Lịch học chưa cập nhật"
          )}</p>
          <div class="d-flex justify-content-between small text-muted">
            <span><i class="fas fa-users me-1"></i>${
              cls.student_count || 0
            } sinh viên</span>
            <span><i class="fas fa-book me-1"></i>${
              cls.semester || "---"
            }</span>
          </div>
          ${
            isSessionOnly
              ? '<p class="text-muted small mt-2"><i class="fas fa-info-circle me-1"></i>Phiên điểm danh đang mở cho lớp này. Bạn chưa được ghi danh chính thức.</p>'
              : ""
          }
        </div>`;
      fragment.appendChild(col);
    });
    this.studentClassList.appendChild(fragment);
  }

  populateStudentClassSelect(classes = []) {
    if (!this.studentClassSelectEl) {
      return;
    }
    const previousValue = this.studentClassSelectEl.value;
    let autoSelectValue = "";
    this.studentClassSelectEl.innerHTML =
      '<option value="">Chọn lớp tín chỉ</option>';

    classes.forEach((cls) => {
      if (!cls) {
        return;
      }
      const option = document.createElement("option");
      const classId = cls.id || cls.credit_class_id;
      option.value = classId || "";
      option.textContent = this.escapeHtml(
        cls.display_name ||
          cls.credit_code ||
          cls.subject_name ||
          `Lớp tín chỉ ${classId || ""}`
      );
      if (cls.active_session) {
        option.dataset.sessionStatus = "open";
        if (!previousValue && !autoSelectValue && option.value) {
          autoSelectValue = option.value;
        }
      }
      if (cls.is_session_only) {
        option.dataset.sessionTemporary = "true";
        option.textContent += " (Phiên đang mở)";
      }
      this.studentClassSelectEl.appendChild(option);
    });

    let targetValue = "";
    if (previousValue) {
      targetValue = previousValue;
    } else if (this.studentPendingSessionClassId) {
      targetValue = String(this.studentPendingSessionClassId);
    } else if (autoSelectValue) {
      targetValue = autoSelectValue;
    }

    if (targetValue) {
      const optionExists = Array.from(this.studentClassSelectEl.options).some(
        (option) => option.value && option.value === targetValue
      );
      if (optionExists) {
        this.studentClassSelectEl.value = targetValue;
        if (
          this.studentPendingSessionClassId &&
          String(this.studentPendingSessionClassId) === targetValue
        ) {
          this.studentPendingSessionClassId = null;
        }
      }
    }
    this.studentSelectedClassId = this.studentClassSelectEl.value;

    if (!previousValue && this.studentClassSelectEl.value) {
      this.validateStudentSession(false);
    }
  }

  showStudentSessionAlert(
    message = "",
    isError = false,
    hide = false,
    isSuccess = false
  ) {
    if (!this.studentSessionAlert) {
      return;
    }
    const alertEl = this.studentSessionAlert;
    if (hide || !message) {
      alertEl.classList.add("d-none");
      alertEl.textContent = "";
      alertEl.classList.remove("alert-danger", "alert-info", "alert-success");
      alertEl.classList.add("alert-info");
      return;
    }
    alertEl.textContent = message;
    alertEl.classList.remove("d-none");
    alertEl.classList.remove("alert-danger", "alert-info", "alert-success");
    if (isSuccess) {
      alertEl.classList.add("alert-success");
    } else {
      alertEl.classList.add(isError ? "alert-danger" : "alert-info");
    }
  }

  setStudentSessionHint(message = "") {
    if (!this.studentSessionHint) {
      return;
    }
    this.studentSessionHint.textContent = message || "";
  }

  toggleStudentCameraCard(show) {
    if (!this.studentCameraWrapper) {
      return;
    }
    this.studentCameraWrapper.classList.toggle("d-none", !show);
  }

  async validateStudentSession(showLoading = true) {
    if (!this.studentClassSelectEl || !this.studentStartBtn) {
      return;
    }
    const classId = Number(this.studentClassSelectEl.value);
    this.studentSelectedClassId = this.studentClassSelectEl.value;
    if (!classId) {
      this.studentSessionState = null;
      this.studentStartBtn.disabled = true;
      this.toggleStudentCameraCard(false);
      this.showStudentSessionAlert(
        "Vui lòng chọn lớp tín chỉ để bắt đầu.",
        false
      );
      return;
    }

    if (showLoading) {
      this.showStudentSessionAlert("Đang kiểm tra phiên điểm danh...", false);
    }
    this.setStudentSessionHint("Đang xác thực phiên với giảng viên...");

    try {
      const payload = await this.api.getSession();
      this.applyStudentSessionState(payload.session || null, {
        skipAutoSelect: true,
        skipClassRefresh: true,
      });
    } catch (error) {
      this.studentSessionState = null;
      this.studentStartBtn.disabled = true;
      this.toggleStudentCameraCard(false);
      this.showStudentSessionAlert(error.message, true);
      this.setStudentSessionHint("Không thể kiểm tra phiên. Thử lại sau.");
    }
  }

  applyStudentSessionState(sessionPayload, options = {}) {
    if (!this.studentPortalEl || !this.studentClassSelectEl) {
      return;
    }
    const { skipAutoSelect = false, skipClassRefresh = false } = options;
    const previousSessionId = this.studentSessionState
      ? this.studentSessionState.id || null
      : null;
    const status = (sessionPayload?.status || "").toLowerCase();
    const isOpen = status === "open";
    const sessionClassId =
      isOpen && sessionPayload?.credit_class_id
        ? Number(sessionPayload.credit_class_id)
        : null;

    if (
      !skipAutoSelect &&
      isOpen &&
      sessionClassId &&
      !this.studentClassSelectEl.value
    ) {
      const matchingOption = Array.from(this.studentClassSelectEl.options).find(
        (option) => option.value && Number(option.value) === sessionClassId
      );
      if (matchingOption) {
        this.studentClassSelectEl.value = matchingOption.value;
      }
    }

    this.studentSelectedClassId = this.studentClassSelectEl.value;
    const selectedClassId = Number(this.studentClassSelectEl.value);

    if (!selectedClassId) {
      this.studentSessionState = null;
      if (this.studentStartBtn) {
        this.studentStartBtn.disabled = true;
      }
      this.toggleStudentCameraCard(false);
      if (isOpen) {
        this.showStudentSessionAlert(
          "Giảng viên đã mở phiên. Hãy chọn lớp tín chỉ phù hợp để tham gia.",
          false
        );
        this.setStudentSessionHint("Chọn lớp có phiên đang mở để bật camera.");
      } else {
        this.showStudentSessionAlert(
          "Vui lòng chọn lớp tín chỉ để kiểm tra phiên điểm danh.",
          false
        );
        this.setStudentSessionHint(
          "Chọn lớp và nhấn Bắt đầu khi phiên được mở."
        );
      }
    } else if (isOpen && sessionClassId === selectedClassId) {
      this.studentSessionState = sessionPayload;
      if (this.studentStartBtn) {
        this.studentStartBtn.disabled = false;
      }
      const expiresText = sessionPayload?.expires_at
        ? this.formatDateTime(sessionPayload.expires_at)
        : "khi giảng viên đóng phiên";
      this.showStudentSessionAlert(
        `Phiên đang mở đến ${expiresText}. Bạn có thể bắt đầu.`,
        false
      );
      this.setStudentSessionHint("Bật camera để điểm danh.");
    } else {
      if (this.studentCameraActive) {
        this.stopStudentCamera(
          "Phiên điểm danh đã đóng hoặc không thuộc lớp bạn."
        );
      } else {
        this.toggleStudentCameraCard(false);
      }
      this.studentSessionState = null;
      if (this.studentStartBtn) {
        this.studentStartBtn.disabled = true;
      }
      const message = isOpen
        ? "Phiên hiện tại thuộc lớp tín chỉ khác."
        : "Lớp này hiện chưa có phiên điểm danh mở.";
      this.showStudentSessionAlert(message, !isOpen);
      this.setStudentSessionHint(
        isOpen
          ? "Chọn đúng lớp hoặc chờ giảng viên của bạn mở phiên."
          : "Liên hệ giảng viên để mở phiên cho lớp bạn."
      );
    }

    const sessionMatchesOption = sessionClassId
      ? Array.from(this.studentClassSelectEl.options).some(
          (option) => option.value && Number(option.value) === sessionClassId
        )
      : false;

    let shouldRefreshClasses = false;
    if (sessionPayload) {
      shouldRefreshClasses = true;
      if (isOpen && sessionClassId && !sessionMatchesOption) {
        this.studentPendingSessionClassId = sessionClassId;
      }
    } else if (previousSessionId) {
      shouldRefreshClasses = true;
    }

    if (
      shouldRefreshClasses &&
      !skipClassRefresh &&
      this.studentPortalEl.dataset.studentId
    ) {
      this.fetchStudentClasses(true);
    }
  }

  async startStudentCamera() {
    if (!this.studentPortalEl || this.studentCameraActive) {
      return;
    }
    await this.validateStudentSession(false);
    if (!this.studentSessionState) {
      return;
    }
    if (!this.studentVideoEl) {
      this.showStudentSessionAlert(
        "Không tìm thấy thành phần camera trong trang.",
        true
      );
      return;
    }

    const classId = this.studentClassSelectEl.value;
    const params = new URLSearchParams({
      action: this.studentCurrentAction || "checkin",
      credit_class_id: classId,
      ts: Date.now().toString(),
    });
    const baseUrl =
      this.studentVideoEl.dataset.feedBase ||
      this.studentVideoEl.src ||
      "/video_feed";
    this.studentVideoEl.src = `${baseUrl}?${params.toString()}`;
    this.studentVideoEl.classList.add("d-none");
    this.toggleStudentCameraCard(true);
    this.studentCameraActive = true;
    if (this.studentStartBtn) {
      this.studentStartBtn.disabled = true;
    }
    if (this.studentStopBtn) {
      this.studentStopBtn.classList.remove("d-none");
    }
    this.showStudentSessionAlert("Đang khởi chạy camera...", false);
  }

  stopStudentCamera(message = "") {
    if (!this.studentPortalEl) {
      return;
    }
    this.studentCameraActive = false;
    if (this.studentVideoEl) {
      this.studentVideoEl.src = "";
      this.studentVideoEl.classList.add("d-none");
    }
    this.toggleStudentCameraCard(false);
    if (this.studentStopBtn) {
      this.studentStopBtn.classList.add("d-none");
    }
    const alertMessage =
      message || "Camera đã tắt. Bạn có thể bắt đầu lại khi cần.";
    this.showStudentSessionAlert(alertMessage, Boolean(message));
    if (this.studentSessionState) {
      this.setStudentSessionHint("Bấm Bắt đầu để tiếp tục phiên.");
    } else {
      this.setStudentSessionHint("Chọn lớp có phiên đang mở để tiếp tục.");
    }
    if (this.studentStartBtn && this.studentSessionState) {
      this.studentStartBtn.disabled = false;
    }
  }

  toggleStudentClassLoading(isLoading) {
    if (this.studentClassLoading) {
      this.studentClassLoading.classList.toggle("d-none", !isLoading);
    }
    if (this.studentClassList) {
      this.studentClassList.classList.toggle("d-none", isLoading);
    }
  }

  showStudentClassEmpty(message = "", isError = false, hide = false) {
    if (!this.studentClassEmpty) {
      return;
    }
    if (hide) {
      this.studentClassEmpty.classList.add("d-none");
      return;
    }
    this.studentClassEmpty.classList.remove("d-none");
    this.studentClassEmpty.innerHTML = message
      ? `<p class="${
          isError ? "text-danger" : "text-muted"
        } mb-0">${message}</p>`
      : this.studentClassEmptyDefault;
  }

  updateStudentSummary(summary = {}) {
    if (this.studentSummaryTotalEl) {
      this.studentSummaryTotalEl.textContent = summary.total_classes || 0;
    }
    if (this.studentSummaryActiveEl) {
      this.studentSummaryActiveEl.textContent = summary.active_sessions || 0;
    }
  }

  async fetchStudentHistory(forceReload = false) {
    if (!this.studentPortalEl || !this.studentPortalEl.dataset.studentId) {
      return;
    }
    this.toggleStudentHistoryLoading(true);
    const studentId = this.studentPortalEl.dataset.studentId;
    try {
      const data = await this.api.getStudentHistory(studentId, {
        limit: 10,
        cacheBust: forceReload,
      });
      this.renderStudentHistory(data.history || []);
    } catch (error) {
      this.renderStudentHistory([]);
      this.showStudentHistoryEmpty(
        error.message || "Không thể tải lịch sử",
        true
      );
    } finally {
      this.toggleStudentHistoryLoading(false);
    }
  }

  renderStudentHistory(history = []) {
    if (!this.studentHistoryList) {
      return;
    }
    this.studentHistoryList.innerHTML = "";
    if (!history.length) {
      this.showStudentHistoryEmpty("Chưa có lịch sử điểm danh.");
      return;
    }
    this.showStudentHistoryEmpty("", false, true);
    const fragment = document.createDocumentFragment();
    history.forEach((item) => {
      const entry = document.createElement("div");
      entry.className = "student-history-item";
      const badgeClass = this.historyStatusClass(item.status);
      entry.innerHTML = `
        <div>
          <strong>${this.escapeHtml(
            item.class_display || item.credit_class_code || "Phiên học"
          )}</strong>
          <div class="history-meta">
            ${this.escapeHtml(
              item.attendance_date || "--"
            )} · ${this.escapeHtml(item.credit_class_code || "---")}
          </div>
          <div class="history-meta">
            Vào: ${
              item.check_in_time ? this.formatTime(item.check_in_time) : "--"
            } · Ra: ${
        item.check_out_time ? this.formatTime(item.check_out_time) : "--"
      }
          </div>
        </div>
        <span class="badge ${badgeClass}">${this.escapeHtml(
        item.status || "--"
      )}</span>`;
      fragment.appendChild(entry);
    });
    this.studentHistoryList.appendChild(fragment);
  }

  historyStatusClass(status) {
    const normalized = (status || "").toLowerCase();
    switch (normalized) {
      case "present":
        return "bg-success";
      case "late":
        return "bg-warning text-dark";
      case "absent":
        return "bg-danger";
      case "excused":
        return "bg-info text-dark";
      default:
        return "bg-secondary";
    }
  }

  toggleStudentHistoryLoading(isLoading) {
    if (this.studentHistoryLoading) {
      this.studentHistoryLoading.classList.toggle("d-none", !isLoading);
    }
    if (this.studentHistoryList) {
      this.studentHistoryList.classList.toggle("d-none", isLoading);
    }
  }

  showStudentHistoryEmpty(message = "", isError = false, hide = false) {
    if (!this.studentHistoryEmpty) {
      return;
    }
    if (hide) {
      this.studentHistoryEmpty.classList.add("d-none");
      return;
    }
    this.studentHistoryEmpty.classList.remove("d-none");
    this.studentHistoryEmpty.innerHTML = message
      ? `<p class="${
          isError ? "text-danger" : "text-muted"
        } mb-0">${message}</p>`
      : this.studentHistoryEmptyDefault;
  }

  async refreshAttendanceList(triggeredByRealtime = false) {
    try {
      const data = await this.api.getTodayAttendance();

      this.renderAttendanceTable(data.data || []);
      this.updateSummarySections(data.checked_in || [], data.checked_out || []);
      if (Object.prototype.hasOwnProperty.call(data, "session")) {
        this.handleSessionEvent(data.session || null);
      }
    } catch (error) {
      const logger = triggeredByRealtime ? console.warn : console.error;
      logger("Error refreshing attendance list:", error);
    }
  }

  renderAttendanceTable(rows) {
    const tbody = document.getElementById("attendance-table-body");
    const countBadge = document.getElementById("attendance-count");
    if (!tbody) {
      return;
    }

    if (!rows.length) {
      tbody.innerHTML = `
        <tr>
          <td colspan="6">
            <div class="text-center py-5">
              <i class="fas fa-user-times fa-3x text-muted mb-3"></i>
              <h5 class="text-muted">Chưa có sinh viên nào điểm danh</h5>
              <p class="text-muted">Hệ thống sẽ tự động nhận diện và điểm danh khi có sinh viên xuất hiện trước camera.</p>
            </div>
          </td>
        </tr>`;
    } else {
      const html = rows
        .map((row, index) => this.createAttendanceRow(row, index))
        .join("");
      tbody.innerHTML = html;
    }

    if (countBadge) {
      countBadge.textContent = rows.length;
    }
  }

  createAttendanceRow(row, index) {
    const duration = this.formatDuration(row.duration_minutes);
    const status = {
      "Đang có mặt": { class: "bg-success", icon: "fa-user-check" },
      "Đã rời": { class: "bg-secondary", icon: "fa-sign-out-alt" },
    }[row.status] || { class: "bg-warning", icon: "fa-user-clock" };
    const rawName = row.full_name || "Không có tên";
    const safeName = this.escapeHtml(rawName);
    const safeId = this.escapeHtml(row.student_id || "N/A");
    const buttonStudentId = this.escapeHtml(row.student_id || "");
    const initials = rawName ? rawName.charAt(0).toUpperCase() : "N";
    const checkInTime = this.formatTime(row.timestamp);
    const checkOutTime = row.checkout_time
      ? this.formatTime(row.checkout_time)
      : "";
    const buttonName = this.escapeHtml(rawName);
    const classBadge = this.renderClassBadge(row);

    return `
      <tr data-student-id="${row.student_id}">
        <td>${index + 1}</td>
        <td>
          <div class="d-flex align-items-center">
            <div class="avatar-sm bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-2">${initials}</div>
            <div>
              <div class="fw-bold">${safeName}</div>
              <small class="text-muted">ID: ${safeId}</small>
              ${classBadge}
            </div>
          </div>
        </td>
        <td>
          <div><i class="fas fa-clock me-1"></i>${checkInTime}</div>
          ${
            checkOutTime
              ? `<small class="text-muted"><i class="fas fa-sign-out-alt me-1"></i>Ra: ${checkOutTime}</small>`
              : ""
          }
        </td>
        <td><span class="badge bg-info">${duration}</span></td>
        <td><span class="badge ${status.class}"><i class="fas ${
      status.icon
    } me-1"></i>${row.status}</span></td>
        <td><button class="btn btn-sm btn-outline-primary js-view-details" data-student-id="${buttonStudentId}" data-student-name="${buttonName}"><i class="fas fa-eye"></i></button></td>
      </tr>`;
  }

  renderClassBadge(row) {
    if (!row || !row.class_display) {
      return "";
    }
    const badgeType =
      (row.class_type || "") === "credit" ? "badge-credit" : "badge-admin";
    const label = this.escapeHtml(row.class_display);
    return `
      <div class="attendance-class-chip mt-1">
        <span class="badge class-badge ${badgeType}">${label}</span>
      </div>`;
  }

  async updateStatistics() {
    const totalEl = document.getElementById("total-students");
    const attendedEl = document.getElementById("attended-students");
    const rateEl = document.getElementById("attendance-rate");
    const avgEl = document.getElementById("avg-duration");

    if (!totalEl && !attendedEl && !rateEl && !avgEl) {
      return;
    }

    try {
      const data = await this.api.getStatistics();
      if (totalEl) {
        totalEl.textContent = data.total_students || 0;
      }
      if (attendedEl) {
        attendedEl.textContent = data.attended_students || 0;
      }
      if (rateEl) {
        rateEl.textContent = `${data.attendance_rate || 0}%`;
      }
      if (avgEl) {
        const avgMins = data.avg_duration_minutes || 0;
        avgEl.textContent = `${Math.floor(avgMins / 60)}h ${avgMins % 60}p`;
      }
    } catch (error) {
      console.error("Error updating statistics:", error);
    }
  }

  async updateActivePresence() {
    try {
      const data = await this.api.getActivePresence();
      const activeList = document.getElementById("active-presence-list");
      if (!activeList) {
        return;
      }
      activeList.innerHTML = "";
      if (data.success && data.count > 0) {
        data.data.forEach((student) => {
          activeList.innerHTML += `<div class="d-flex justify-content-between align-items-center mb-2"><span>${student.name}</span><span class="badge bg-success">Đang có mặt</span></div>`;
        });
      } else {
        activeList.innerHTML = `<div class="text-center text-muted"><i class="fas fa-user-clock fa-2x mb-2"></i><p>Chưa có sinh viên nào.</p></div>`;
      }
    } catch (error) {
      console.error("Error updating active presence:", error);
    }
  }

  updateSummarySections(checkedIn = [], checkedOut = []) {
    const updateList = ({
      countId,
      listId,
      items,
      timeKey,
      emptyLabel,
      badgeClass,
      badgeText,
      timeLabel,
    }) => {
      const countEl = document.getElementById(countId);
      const listEl = document.getElementById(listId);
      if (countEl) {
        countEl.textContent = items.length;
      }
      if (!listEl) {
        return;
      }

      if (!items.length) {
        listEl.innerHTML = `<div class="list-group-item text-center text-muted">${emptyLabel}</div>`;
        return;
      }

      listEl.innerHTML = items
        .map((item) => {
          const name = this.escapeHtml(
            item.full_name || item.student_name || "Không rõ"
          );
          const timeValue = this.formatTime(item[timeKey] || item.timestamp);
          return `
            <div class="list-group-item d-flex justify-content-between align-items-center">
              <div>
                <div class="fw-bold">${name}</div>
                <small class="text-muted">${timeLabel}: ${timeValue}</small>
              </div>
              <span class="badge ${badgeClass}">${badgeText}</span>
            </div>`;
        })
        .join("");
    };

    updateList({
      countId: "checked-in-count",
      listId: "checked-in-list",
      items: checkedIn,
      timeKey: "timestamp",
      emptyLabel: "Chưa có sinh viên nào check-in",
      badgeClass: "bg-success",
      badgeText: "Đang có mặt",
      timeLabel: "Vào",
    });

    updateList({
      countId: "checked-out-count",
      listId: "checked-out-list",
      items: checkedOut,
      timeKey: "checkout_time",
      emptyLabel: "Chưa có sinh viên nào checkout",
      badgeClass: "bg-secondary",
      badgeText: "Đã checkout",
      timeLabel: "Ra",
    });
  }

  initAttendanceSessionPanel() {
    const card = document.getElementById("credit-session-card");
    if (!card) {
      return;
    }
    this.sessionElements = {
      card,
      select: document.getElementById("credit-class-select"),
      openBtn: document.getElementById("open-session-btn"),
      closeBtn: document.getElementById("close-session-btn"),
      formContainer: document.getElementById("session-form-container"),
      activeContainer: document.getElementById("active-session-container"),
      statusPill: document.getElementById("session-status-pill"),
      alert: document.getElementById("session-alert"),
      className: document.getElementById("session-class-name"),
      classCode: document.getElementById("session-class-code"),
      openedAt: document.getElementById("session-opened-at"),
      deadline: document.getElementById("session-deadline"),
      countdown: document.getElementById("session-countdown"),
    };

    if (this.sessionElements.openBtn) {
      this.sessionElements.openBtn.addEventListener("click", () =>
        this.handleOpenSession()
      );
    }
    if (this.sessionElements.closeBtn) {
      this.sessionElements.closeBtn.addEventListener("click", () =>
        this.handleCloseSession()
      );
    }

    window.addEventListener("beforeunload", () => {
      this.stopSessionCountdown();
      if (this.sessionPollInterval) {
        clearInterval(this.sessionPollInterval);
        this.sessionPollInterval = null;
      }
    });
    this.fetchCreditClassesForSession();
    this.refreshSessionState(true);
    this.sessionPollInterval = setInterval(
      () => this.refreshSessionState(true),
      15000
    );
  }

  async fetchCreditClassesForSession(forceReload = false) {
    if (!this.sessionElements.select) {
      return;
    }
    if (!forceReload && this.creditClassOptions.length) {
      this.populateCreditClassSelect();
      return;
    }
    if (!forceReload && this.creditClassPromise) {
      await this.creditClassPromise;
      return;
    }

    const loader = (async () => {
      try {
        const response = await fetch(`/api/credit-classes?t=${Date.now()}`);
        const payload = await response.json();
        this.creditClassOptions = payload.success ? payload.data || [] : [];
      } catch (error) {
        console.error("Error loading credit classes:", error);
        this.creditClassOptions = [];
      } finally {
        this.creditClassPromise = null;
        this.populateCreditClassSelect();
      }
    })();

    if (!forceReload) {
      this.creditClassPromise = loader;
    }

    return loader;
  }

  populateCreditClassSelect() {
    const select = this.sessionElements.select;
    if (!select) {
      return;
    }
    select.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = this.creditClassOptions.length
      ? "-- Chọn lớp tín chỉ --"
      : "-- Chưa có lớp tín chỉ --";
    select.appendChild(placeholder);

    this.creditClassOptions.forEach((cls) => {
      const option = document.createElement("option");
      option.value = cls.id;
      option.textContent =
        cls.display_name ||
        [cls.subject_name, cls.credit_code].filter(Boolean).join(" · ");
      if (cls.student_count) {
        option.textContent += ` (${cls.student_count} SV)`;
      }
      select.appendChild(option);
    });
  }

  handleSessionEvent(sessionPayload) {
    if (this.sessionElements.card) {
      this.updateSessionUI(sessionPayload);
    }
    this.applyStudentSessionState(sessionPayload || null);
  }

  async handleOpenSession() {
    if (!this.sessionElements.select || !this.sessionElements.openBtn) {
      return;
    }
    const classId = this.sessionElements.select.value;
    if (!classId) {
      this.setSessionAlert("Vui lòng chọn lớp tín chỉ", "warning");
      return;
    }

    this.setSessionAlert("");
    this.toggleButtonLoading(
      this.sessionElements.openBtn,
      true,
      "Đang mở phiên..."
    );
    try {
      const payload = await this.api.openSession(classId);
      this.setSessionAlert("Đã mở phiên điểm danh thành công", "success");
      this.handleSessionEvent(payload.session || null);
      this.refreshAttendanceList(true);
    } catch (error) {
      this.setSessionAlert(error.message || "Không thể mở phiên điểm danh");
    } finally {
      this.toggleButtonLoading(this.sessionElements.openBtn, false);
    }
  }

  async handleCloseSession() {
    if (!this.sessionElements.closeBtn) {
      return;
    }
    this.toggleButtonLoading(
      this.sessionElements.closeBtn,
      true,
      "Đang đóng phiên..."
    );
    this.setSessionAlert("");
    try {
      const payload = await this.api.closeSession();
      this.setSessionAlert("Phiên điểm danh đã được đóng", "info");
      this.handleSessionEvent(payload.session || null);
      this.refreshAttendanceList(true);
    } catch (error) {
      this.setSessionAlert(error.message || "Không thể đóng phiên");
    } finally {
      this.toggleButtonLoading(this.sessionElements.closeBtn, false);
    }
  }

  async refreshSessionState(silent = false) {
    if (!this.sessionElements.card) {
      return;
    }
    try {
      const payload = await this.api.getSession();
      this.handleSessionEvent(payload.session || null);
    } catch (error) {
      if (!silent) {
        this.setSessionAlert(error.message || "Không thể tải trạng thái phiên");
      }
    }
  }

  updateSessionUI(sessionPayload) {
    if (!this.sessionElements.statusPill) {
      return;
    }
    const hasActiveSession =
      sessionPayload && (sessionPayload.status || "") === "open";
    this.activeSession = hasActiveSession ? sessionPayload : null;
    this.sessionElements.statusPill.textContent = hasActiveSession
      ? "Đang mở"
      : "Đang đóng";
    this.sessionElements.statusPill.className = `badge ${
      hasActiveSession ? "bg-success" : "bg-secondary"
    }`;

    if (this.sessionElements.formContainer) {
      this.sessionElements.formContainer.classList.toggle(
        "d-none",
        !!hasActiveSession
      );
    }
    if (this.sessionElements.activeContainer) {
      this.sessionElements.activeContainer.classList.toggle(
        "d-none",
        !hasActiveSession
      );
    }

    if (!hasActiveSession) {
      this.stopSessionCountdown();
      return;
    }

    this.sessionElements.className.textContent =
      sessionPayload.class_name || sessionPayload.class_code || "---";
    this.sessionElements.classCode.textContent = sessionPayload.class_code
      ? `Mã lớp: ${sessionPayload.class_code}`
      : "";
    this.sessionElements.openedAt.textContent = sessionPayload.opened_at
      ? this.formatDateTime(sessionPayload.opened_at)
      : "--";
    const expiresAt =
      sessionPayload.expires_at ||
      sessionPayload.checkin_deadline ||
      sessionPayload.checkout_deadline;
    this.sessionElements.deadline.textContent = expiresAt
      ? this.formatDateTime(expiresAt)
      : "--";
    this.startSessionCountdown();
  }

  setSessionAlert(message = "", type = "danger") {
    if (!this.sessionElements.alert) {
      return;
    }
    if (!message) {
      this.sessionElements.alert.classList.add("d-none");
      this.sessionElements.alert.textContent = "";
      this.sessionElements.alert.className = "alert alert-danger d-none";
      return;
    }
    this.sessionElements.alert.textContent = message;
    this.sessionElements.alert.className = `alert alert-${type}`;
  }

  startSessionCountdown() {
    this.stopSessionCountdown();
    this.updateSessionCountdown();
    this.sessionCountdownInterval = setInterval(
      () => this.updateSessionCountdown(),
      1000
    );
  }

  stopSessionCountdown() {
    if (this.sessionCountdownInterval) {
      clearInterval(this.sessionCountdownInterval);
      this.sessionCountdownInterval = null;
    }
    if (this.sessionElements.countdown) {
      this.sessionElements.countdown.textContent = "--:--";
    }
  }

  updateSessionCountdown() {
    if (!this.sessionElements.countdown || !this.activeSession) {
      return;
    }
    const deadline =
      this.activeSession.expires_at ||
      this.activeSession.checkin_deadline ||
      this.activeSession.checkout_deadline;
    if (!deadline) {
      this.sessionElements.countdown.textContent = "--:--";
      return;
    }
    const diffSeconds = Math.max(
      Math.floor((Date.parse(deadline) - Date.now()) / 1000),
      0
    );
    const minutes = String(Math.floor(diffSeconds / 60)).padStart(2, "0");
    const seconds = String(diffSeconds % 60).padStart(2, "0");
    this.sessionElements.countdown.textContent = `${minutes}:${seconds}`;
    if (diffSeconds <= 0) {
      this.stopSessionCountdown();
      this.refreshSessionState(true);
    }
  }

  toggleButtonLoading(button, isLoading, loadingText) {
    if (!button) {
      return;
    }
    if (isLoading) {
      button.dataset.originalText =
        button.dataset.originalText || button.innerHTML;
      button.disabled = true;
      button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${
        loadingText || "Đang xử lý..."
      }`;
    } else {
      button.disabled = false;
      if (button.dataset.originalText) {
        button.innerHTML = button.dataset.originalText;
        delete button.dataset.originalText;
      }
    }
  }

  initDetailModal() {
    this.detailModalEl = document.getElementById("attendanceDetailModal");
    if (!this.detailModalEl || typeof bootstrap === "undefined") {
      return;
    }
    this.detailModal = new bootstrap.Modal(this.detailModalEl);
    this.detailElements = {
      alert: document.getElementById("attendance-detail-alert"),
      name: document.getElementById("detail-student-name"),
      id: document.getElementById("detail-student-id"),
      className: document.getElementById("detail-student-class"),
      lastCheckIn: document.getElementById("detail-last-checkin"),
      lastCheckOut: document.getElementById("detail-last-checkout"),
      totalSessions: document.getElementById("detail-total-sessions"),
      currentStatus: document.getElementById("detail-current-status"),
      historyBody: document.getElementById("attendance-history-body"),
    };
  }

  bindDetailHandlers() {
    document.addEventListener("click", (event) => {
      const detailBtn = event.target.closest(".js-view-details");
      if (!detailBtn) {
        return;
      }
      const studentId = detailBtn.getAttribute("data-student-id");
      const studentName = detailBtn.getAttribute("data-student-name") || "N/A";
      this.openAttendanceDetail(studentId, studentName);
    });
  }

  async openAttendanceDetail(studentId, fallbackName) {
    if (!this.detailModal) {
      console.warn("Attendance detail modal is not configured");
      return;
    }
    if (!studentId) {
      alert("Không tìm thấy mã sinh viên cho bản ghi này");
      return;
    }

    this.updateDetailHeader(fallbackName || "N/A", studentId, null);
    this.setDetailAlert();
    this.setDetailLoading(true);
    this.detailModal.show();

    try {
      const data = await this.api.getStudentHistory(studentId, {
        cacheBust: true,
      });
      this.updateDetailHeader(
        data.student_name || fallbackName,
        data.student_id || studentId,
        data.class_name
      );
      this.updateDetailSummary(data.summary || {});
      this.renderAttendanceHistory(data.history || []);
    } catch (error) {
      this.renderAttendanceHistory([]);
      this.setDetailAlert(error.message || "Không thể tải dữ liệu");
    } finally {
      this.setDetailLoading(false);
    }
  }

  updateDetailHeader(name, studentId, className) {
    if (!this.detailElements.name) {
      return;
    }
    this.detailElements.name.textContent = name || "---";
    if (this.detailElements.id) {
      this.detailElements.id.textContent = `MSSV: ${studentId || "---"}`;
    }
    if (this.detailElements.className) {
      this.detailElements.className.textContent = className || "Chưa rõ lớp";
    }
  }

  updateDetailSummary(summary = {}) {
    if (!this.detailElements.totalSessions) {
      return;
    }
    this.detailElements.totalSessions.textContent = summary.total_sessions || 0;
    if (this.detailElements.lastCheckIn) {
      this.detailElements.lastCheckIn.textContent = summary.last_check_in
        ? this.formatDateTime(summary.last_check_in)
        : "--";
    }
    if (this.detailElements.lastCheckOut) {
      this.detailElements.lastCheckOut.textContent = summary.last_check_out
        ? this.formatDateTime(summary.last_check_out)
        : "--";
    }
    if (this.detailElements.currentStatus) {
      this.detailElements.currentStatus.textContent =
        summary.current_status || "--";
      const badgeClass = summary.status_class || "bg-info";
      this.detailElements.currentStatus.className = `badge ${badgeClass}`;
    }
  }

  renderAttendanceHistory(history = []) {
    if (!this.detailElements.historyBody) {
      return;
    }
    if (!history.length) {
      this.detailElements.historyBody.innerHTML =
        '<tr><td colspan="5" class="text-center text-muted">Không có dữ liệu lịch sử.</td></tr>';
      return;
    }

    const rows = history
      .map((item) => {
        const dateLabel = this.escapeHtml(item.attendance_date || "--");
        const checkIn = item.check_in_time
          ? this.formatDateTime(item.check_in_time)
          : "--";
        const checkOut = item.check_out_time
          ? this.formatDateTime(item.check_out_time)
          : "--";
        const notes = this.escapeHtml(item.notes || "--");
        return `
          <tr>
            <td>${dateLabel}</td>
            <td>${checkIn}</td>
            <td>${checkOut}</td>
            <td>${this.formatDuration(item.duration_minutes)}</td>
            <td>${notes}</td>
          </tr>`;
      })
      .join("");

    this.detailElements.historyBody.innerHTML = rows;
  }

  setDetailLoading(isLoading) {
    if (!this.detailElements.historyBody) {
      return;
    }
    if (isLoading) {
      this.detailElements.historyBody.innerHTML =
        '<tr><td colspan="5" class="text-center text-muted">Đang tải...</td></tr>';
    }
  }

  setDetailAlert(message = "") {
    if (!this.detailElements.alert) {
      return;
    }
    if (!message) {
      this.detailElements.alert.classList.add("d-none");
      this.detailElements.alert.textContent = "";
      return;
    }
    this.detailElements.alert.textContent = message;
    this.detailElements.alert.classList.remove("d-none");
  }

  async fetchJson(url, options) {
    const response = await fetch(url, options);
    let payload = {};
    try {
      payload = await response.json();
    } catch (error) {
      payload = {};
    }
    const failed = !response.ok || (payload && payload.success === false);
    if (failed) {
      const message =
        (payload && (payload.message || payload.error)) ||
        response.statusText ||
        "Không thể tải dữ liệu";
      throw new Error(message);
    }
    return payload;
  }

  formatDuration(minutes) {
    const totalMinutes = Number(minutes);
    if (Number.isNaN(totalMinutes) || totalMinutes < 0) {
      return "0 phút";
    }
    if (totalMinutes >= 60) {
      const hours = Math.floor(totalMinutes / 60);
      const mins = Math.round(totalMinutes % 60);
      return `${hours}h ${mins}p`;
    }
    return `${Math.round(totalMinutes)} phút`;
  }

  formatTime(value) {
    if (!value || typeof value !== "string") {
      return "N/A";
    }
    if (value.length >= 19) {
      return value.substring(11, 19);
    }
    return value;
  }

  escapeHtml(value = "") {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  formatDateTime(value) {
    if (!value) {
      return "--";
    }
    const dateObj = new Date(value);
    if (Number.isNaN(dateObj.getTime())) {
      return value;
    }
    return `${dateObj.toLocaleDateString("vi-VN")} ${dateObj.toLocaleTimeString(
      "vi-VN"
    )}`;
  }
}
