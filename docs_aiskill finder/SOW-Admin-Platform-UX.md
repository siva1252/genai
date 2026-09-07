# Statement of Work: SkillSync / RByte.ai — Admin Platform & Student UX

**Client:** TBD  
**Vendor / Delivery team:** SkillSync / RByte.ai product team  
**Document version:** 1.0  
**Date:** 2026-09-03  
**Status:** Draft  
**Related docs:** `project.md`, `SkillSync_API_Spec.md`, `README.md`

---

## 1. Overview

This Statement of Work defines delivery of the **admin platform management surface**, **one-time credit purchase plans**, **multi-provider authentication**, and the **student UX navigation** for SkillSync (RByte.ai assessment product).

The goal is a coherent product journey from sign-in through assessment, reports, learning path, and resume writer — with an owner-facing admin console for audit and KPIs — matching the architecture and modules already described in `project.md`.

---

## 2. Objectives

- Give students a working sidebar and pages for **Dashboard, Assessments, Reports, Learning Path, Resume Writer, and Settings**.
- Ensure **Learning Path** and **Reports** use the **latest completed assessment**, not empty session state.
- Generate a learning path **once** via LangGraph; subsequent visits **load the saved path** (`GET /recommendation/path`).
- Provide platform owners a **read-only admin dashboard** for users, assessments, payments, and logs.
- Support **Google, Email, and LinkedIn** sign-in and **Razorpay** one-time credit packs.

### Success measures

- Sidebar routes open the correct page and do not bounce incorrectly.
- Users with completed assessments see Reports and Learning Path content without being told to “start an assessment”.
- Opening Learning Path after a path already exists does not re-run the full “Building your path” agent flow.
- Admin console is available only to `is_admin` users and remains read-only by default.

---

## 3. Scope

### 3.1 In scope

#### A. Student UX (priority — Medium roadmap item in `project.md`)

| Area | Requirement |
|------|-------------|
| Shared app shell | Persistent sidebar on student pages with active-state highlighting |
| Dashboard | Credits, stats, assessment history, start new assessment |
| Assessments | Opens start-assessment flow from sidebar |
| Reports | Dedicated `/reports` page listing completed assessments; open report drawer/detail |
| Learning Path | Load path for latest assessment; show saved path if present; generate only when missing |
| Resume Writer | Available from sidebar; uses latest benchmark when available |
| Settings | Profile / settings at `/profile` (extracted skills and profile fields) |
| Session continuity | Persist `session_id` + `benchmark_id` from assessment history so post-assessment modules work |

#### B. Authentication

- Google OAuth sign-in  
- Email register / login  
- LinkedIn OAuth (login + callback + frontend complete flow)  
- JWT session for API calls  

#### C. Credits & payments

| Plan | Price (INR) | Credits |
|------|-------------|---------|
| Basic | ₹49 | 1 assessment |
| Pro | ₹149 | 3 assessments |
| Placement Pack | ₹299 | Unlimited (999 credits) |

- Razorpay create-order + verify  
- Credit balance on dashboard (`GET /payments/me`)  

#### D. Admin platform (read-only by default)

- Summary KPIs, users, logins, assessments, resumes, payments, audit log, courses, questions  
- Access gated by `is_admin` / `ADMIN_EMAILS`  
- No student-facing admin; no manual student credit edits in production read-only mode  

#### E. Recommendation API behaviour

- `POST /recommendation/generate` — create path for an assessment; if a path already exists for that `session_id`, return it  
- `GET /recommendation/path` — return saved path (optional `session_id`); 404 if none  

### 3.2 Out of scope

- Multi-language resume support  
- Placement-cell analytics product (beyond current admin KPIs)  
- Streaming LLM responses for live evaluation feedback  
- Human-reviewed question bank as GPT fallback  
- Admin write/edit of users, credits, or courses when `ADMIN_READ_ONLY=true`  
- Mobile native apps  
- Invented commercials / SLAs not listed in Section 8  

---

## 4. Deliverables

| ID | Deliverable | Description | Format |
|----|-------------|-------------|--------|
| D1 | Student AppShell | Shared sidebar + mobile nav across student routes | Next.js components |
| D2 | Reports page | `/reports` history + report viewer | Frontend page |
| D3 | Learning Path UX | Latest-assessment binding + load-before-generate | Frontend + API |
| D4 | Assessment history API fields | Include `benchmark_id` (and session) for client hydration | FastAPI |
| D5 | Recommendation idempotency | Generate returns existing path when present | FastAPI |
| D6 | Settings / Profile | Settings entry via `/profile` under AppShell | Frontend |
| D7 | Auth providers | Google, Email, LinkedIn as in product | Backend + Frontend |
| D8 | Credit purchase UX | Plans + Razorpay modal on dashboard | Frontend + Payments API |
| D9 | Admin console | Read-only admin dashboard per Module 9 | Frontend + `/admin/*` |
| D10 | This SOW | Living requirements for admin + student UX | `docs/SOW-Admin-Platform-UX.md` |

---

## 5. Timeline and milestones

| Milestone | Description | Target date |
|-----------|-------------|-------------|
| M1 | Student sidebar + Assessments / Reports / Settings wiring | TBD |
| M2 | Learning Path load-from-latest-assessment + no redundant regenerate | TBD |
| M3 | Auth + credits polish (Google / Email / LinkedIn / Razorpay) | TBD |
| M4 | Admin read-only console acceptance | TBD |
| M5 | Sign-off against acceptance criteria (Section 9) | TBD |

---

## 6. Roles and responsibilities

### Client

- Provide brand/copy feedback and acceptance sign-off  
- Supply OAuth client IDs, Razorpay keys, and OpenAI / Azure secrets for each environment  
- Confirm admin owner emails for `ADMIN_EMAILS`  

### Vendor

- Implement in-scope student UX, recommendation persistence, auth, payments, and admin console  
- Keep `project.md` and this SOW aligned when behaviour changes  
- Deliver against acceptance criteria and demo the student journey end-to-end  

---

## 7. Assumptions and dependencies

- Core GenAI modules (profile, benchmark, assessment, evaluation, gaps, LangGraph, resume writer) remain available as in `project.md`  
- PostgreSQL holds `learning_paths` with `session_id` / `benchmark_id`  
- Frontend uses JWT + session storage; after login, dashboard/history can hydrate latest assessment  
- OpenAI / Azure keys are valid in the target environment  
- Docker or local Postgres + Redis available for demo  

---

## 8. Commercials

- Engagement model: TBD  
- Estimate / budget: TBD  
- Payment schedule: TBD  

*(Product credit prices above are **end-user** plan prices, not vendor engagement fees.)*

---

## 9. Acceptance criteria

### 9.1 Student navigation

- [ ] Sidebar items **Dashboard, Assessments, Learning Path, Resume Writer, Reports, Settings** each navigate to a real route or intended action  
- [ ] **Learning Path** does not redirect to Dashboard when the user has completed assessments  
- [ ] **Reports** does not redirect to Assessments when history exists; empty state only when history is empty  
- [ ] **Settings** opens profile/settings without forcing an unexpected onboarding bounce when a profile exists  

### 9.2 Assessments & reports

- [ ] User with N completed assessments sees them on Dashboard and Reports  
- [ ] Selecting a report opens score / detail view for that session  
- [ ] Assessments sidebar action opens the start-assessment flow  

### 9.3 Learning Path

- [ ] Path is tied to the **latest completed assessment** (`session_id` + `benchmark_id`)  
- [ ] First visit with no saved path may show generation progress once  
- [ ] Later visits load `GET /recommendation/path` (or generate short-circuits to existing) and **show the path without re-running the full agent UI**  
- [ ] Empty “start an assessment” state appears only when the user has **no** completed assessments / required IDs  

### 9.3b Assessment fairness & retakes

- [ ] During the 10 questions, **no** correct/wrong reveal or explanations  
- [ ] After results, each question shows correct vs wrong and **why**  
- [ ] Each assessment includes **at least one coding question**  
- [ ] Retakes do not reuse prior bank questions; difficulty scales with prior count + score  
- [ ] Manual/custom target role drives benchmark and question generation tech stack  

### 9.3c Resume edit

- [ ] After generate, student can **Edit** summary/skills/experience/etc.  
- [ ] PDF download uses the **edited** content

### 9.4 Resume Writer

- [ ] Reachable from sidebar  
- [ ] Uses available benchmark context when present; clear empty state otherwise  

### 9.5 Auth & credits

- [ ] Google, Email, and LinkedIn sign-in paths work per environment config  
- [ ] Credit balance displays; purchase flow uses listed one-time plans  

### 9.6 Admin

- [ ] Non-admin cannot access admin console  
- [ ] Admin can view summary, users, assessments, payments, and related read endpoints  
- [ ] With read-only mode on, mutating admin actions are blocked (403)  

### Sign-off

- Owner: TBD  
- Process: Demo against checklist above + written acceptance  

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Session storage missing `benchmark_id` after dashboard load | Learning Path / Resume Writer think user never assessed | Hydrate from `/assessment/history` including `benchmark_id` |
| Path regenerated on every visit | Slow UX, extra LLM cost | Prefer GET path; make POST generate idempotent per session |
| OAuth / Razorpay misconfiguration | Auth or purchase fails in demo | Env checklist in `.env.example`; smoke test before demo |
| Admin emails wrong | Owners locked out / students elevated | Document `ADMIN_EMAILS`; verify `is_admin` in DB |
| Missing SOW previously | UX scope unclear | This document + keep `project.md` Related Documents link |

---

## 11. Change control

Changes to scope, timeline, or commercials require written agreement and may adjust cost/schedule.

---

## 12. Implementation notes (current codebase alignment)

These items are the intended implementation of this SOW in the SkillSync repo:

| SOW item | Implementation |
|----------|----------------|
| Shared student nav | `frontend/src/components/layout/AppShell.tsx` |
| Reports | `frontend/src/app/reports/page.tsx` |
| Learning Path load-first | `frontend/src/app/learning-path/page.tsx` + `GET/POST /recommendation/*` |
| History hydration | `benchmark_id` on `/assessment/history`; `persistLatestAssessment` in `frontend/src/lib/session.ts` |
| Idempotent generate | `backend/app/api/routes/recommendation.py` returns existing path for `session_id` |
| Settings | `/profile` under AppShell |
| Admin | `/admin` + Module 9 APIs |

---

## Open questions

- Client legal name and commercial terms for this SOW?  
- Target dates for M1–M5?  
- Should a **new** assessment always force a **new** learning path, or keep showing the previous path until the user clicks “Rebuild”?  
- Confirm whether Settings is profile-only for v1 or includes notification/password preferences  

---

*Referenced by `project.md` → Related Documents → Admin SOW (`docs/SOW-Admin-Platform-UX.md`).*
