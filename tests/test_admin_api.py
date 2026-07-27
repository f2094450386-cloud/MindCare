import json
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import router
from app.core.database import Base, get_db
from app.core.security import current_user, require_admin
from app.models.entities import AgentRunTrace, ToolAuditRecord, UserAccount


class AdminTraceAuditApiTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.admin = UserAccount(
            username="admin-api",
            display_name="Admin",
            password_hash="unused",
            roles_csv="ROLE_ADMIN,ROLE_USER",
        )
        self.db.add(self.admin)
        self.db.commit()
        self.db.refresh(self.admin)
        trace = AgentRunTrace(
            user_id=self.admin.id,
            session_id=999,
            intent="CHAT",
            risk_level="LOW",
            original_input="hello",
            sanitized_input="hello",
            memory_brief="none",
            agent_steps_json=json.dumps([{"kind": "agent_event", "type": "FINAL_ACCEPTED"}]),
            retrieved_knowledge_json="[]",
            response_messages_json="[]",
            assessment_json="{}",
        )
        audit = ToolAuditRecord(
            tool_name="EXCEL_REPORT",
            policy="EXCEL_REPORT",
            allowed=True,
            status="SUCCESS",
            reason="允许执行",
            payload="{}",
        )
        self.db.add_all([trace, audit])
        self.db.commit()

        app = FastAPI()
        app.include_router(router)

        def override_db():
            yield self.db

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[current_user] = lambda: self.admin
        app.dependency_overrides[require_admin] = lambda: self.admin
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.db.close()

    def test_admin_trace_and_audit_endpoints_return_service_methods(self):
        traces = self.client.get("/api/admin/agent-traces")
        audits = self.client.get("/api/admin/tool-audits")

        self.assertEqual(traces.status_code, 200)
        self.assertEqual(traces.json()[0]["agentSteps"][0]["kind"], "agent_event")
        self.assertEqual(audits.status_code, 200)
        self.assertEqual(audits.json()[0]["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
