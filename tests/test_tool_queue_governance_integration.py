import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.database import Base
from app.core.enums import EmotionLabel, IntentType, RiskLevel, ToolJobKind, ToolJobStatus
from app.models.entities import PsychologicalReport, ToolAuditRecord, ToolJob, UserAccount, ChatSession
from app.services.tool_queue import ToolQueueWorker


class ToolQueueGovernanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.db = self.Session()
        self.user = UserAccount(
            username="queue-user",
            display_name="Queue User",
            password_hash="unused",
            roles_csv="ROLE_USER",
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)
        self.session = ChatSession(public_id="queue-session", user_id=self.user.id, title="queue")
        self.db.add(self.session)
        self.db.commit()
        self.db.refresh(self.session)
        self.settings = Settings(
            database_url="sqlite://",
            ai_provider="mock",
            knowledge_vector_enabled=False,
            alert_email_delivery_mode="log",
        )
        self.worker = ToolQueueWorker(self.settings)

    def tearDown(self):
        self.worker.stop()
        self.db.close()
        self.engine.dispose()

    def _report(self, risk: RiskLevel):
        report = PsychologicalReport(
            user_id=self.user.id,
            session_id=self.session.id,
            content="integration",
            intent=IntentType.RISK.value if risk == RiskLevel.HIGH else IntentType.CONSULT.value,
            emotion=EmotionLabel.HIGH_RISK.value if risk == RiskLevel.HIGH else EmotionLabel.NORMAL.value,
            emotion_score=4.0 if risk == RiskLevel.HIGH else 0.0,
            risk_level=risk.value,
            confidence=0.95,
            summary="integration",
        )
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        return report

    def test_worker_audits_allowed_execution_and_blocks_disallowed_execution(self):
        allowed_report = self._report(RiskLevel.HIGH)
        allowed_job = ToolJob(
            report_id=allowed_report.id,
            kind=ToolJobKind.EXCEL_REPORT.value,
            status=ToolJobStatus.RUNNING.value,
            attempts=0,
            max_attempts=1,
        )
        blocked_report = self._report(RiskLevel.LOW)
        blocked_job = ToolJob(
            report_id=blocked_report.id,
            kind=ToolJobKind.CASE_CREATE.value,
            status=ToolJobStatus.RUNNING.value,
            attempts=0,
            max_attempts=1,
        )
        self.db.add_all([allowed_job, blocked_job])
        self.db.commit()
        self.db.refresh(allowed_job)
        self.db.refresh(blocked_job)

        with patch("app.services.tool_queue.SessionLocal", self.Session):
            with patch.object(self.worker, "_execute") as execute:
                self.worker._run_job(allowed_job.id)
                execute.assert_called_once()
            with patch.object(self.worker, "_execute") as execute:
                self.worker._run_job(blocked_job.id)
                execute.assert_not_called()

        allowed_audit = self.db.query(ToolAuditRecord).filter(ToolAuditRecord.job_id == allowed_job.id).one()
        blocked_audit = self.db.query(ToolAuditRecord).filter(ToolAuditRecord.job_id == blocked_job.id).one()
        self.assertTrue(allowed_audit.allowed)
        self.assertEqual(allowed_audit.status, "SUCCESS")
        self.assertFalse(blocked_audit.allowed)
        self.assertEqual(blocked_audit.status, "BLOCKED")


if __name__ == "__main__":
    unittest.main()
