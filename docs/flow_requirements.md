# Attendance Flow Constraints

- **Session management**

  - Only teachers (role `teacher`) may open attendance sessions—and only for credit classes assigned to them—while teachers or admins may still close an active session.
  - Each credit class is owned by exactly one teacher, but a teacher can manage multiple credit classes.
  - Administrative classes remain one-to-one with students for reporting; credit classes are used for attendance.

- **Student attendance journey**

  - A student account is linked to exactly one administrative class but may enroll in multiple credit classes via `credit_class_students`.
  - After logging in, the camera must stay hidden until the student selects:
    1. Attendance action (`check-in` or `check-out`).
    2. The credit class corresponding to an open session.
  - Both check-in and check-out flows follow the same pattern: pick action → pick credit class → show camera.

- **Face verification**

  - When a student initiates attendance, the camera may only accept the face that matches the logged-in account.
  - If the recognition system detects a different person (even if their face exists in the DB), the attempt is rejected.

- **Teacher experience**

  - Teachers see only their assigned credit classes in dashboards.
  - Opening a session requires choosing one of their credit classes; the session duration defaults but can be overridden.

- **General notes**
  - Credit classes do not live inside the student record; relationships are stored in `credit_class_students`.
  - SSE notifications and existing attendance summaries must continue working after the flow change.
