import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import ValidationError

from support_agent import (
    create_support_graph,
    load_knowledge_base,
    lookup_order,
    retrieve_documents,
)


class FakeModel:
    def invoke(self, messages):
        return AIMessage(content="Grounded response.")


class SupportAgentTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"title": "Returns", "content": "Unused items can be returned within 30 days."},
            {"title": "Shipping", "content": "Standard shipping takes 3 to 5 business days."},
        ]

    def test_retrieval_ranks_matching_documents_and_ignores_unrelated_ones(self):
        results = retrieve_documents("How long to return an unused item?", self.documents)

        self.assertEqual([document["title"] for document in results], ["Returns"])
        self.assertEqual(retrieve_documents("unrelated question", self.documents), [])

    def test_knowledge_base_rejects_documents_missing_required_fields(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('[{"title": "Missing content"}]', encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_knowledge_base(path)

    def test_knowledge_question_uses_retrieved_context(self):
        graph = create_support_graph(llm=FakeModel(), documents=self.documents)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Can I return an unused item?")]},
            config={"configurable": {"thread_id": "knowledge-test"}},
        )

        self.assertEqual(result["intent"], "knowledge")
        self.assertEqual(result["documents"][0]["title"], "Returns")
        self.assertEqual(result["response"], "Grounded response.")

    def test_order_lookup_uses_order_id(self):
        looked_up = []
        graph = create_support_graph(
            llm=FakeModel(),
            documents=self.documents,
            order_lookup=lambda order_id: looked_up.append(order_id) or "Status: shipped",
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="Where is order #ABCD-1234?")]},
            config={"configurable": {"thread_id": "order-test"}},
        )

        self.assertEqual(looked_up, ["ABCD-1234"])
        self.assertEqual(result["intent"], "order")
        self.assertEqual(result["order_result"], "Status: shipped")

    @patch.dict(
        "os.environ",
        {"ORDER_API_URL": "https://orders.example/api/orders", "ORDER_API_TOKEN": "test-token"},
    )
    @patch("support_agent.urllib.request.urlopen")
    def test_order_api_uses_bearer_auth_and_returns_only_supported_fields(self, urlopen):
        response = MagicMock()
        response.__enter__.return_value = io.BytesIO(
            b'{"status":"shipped","tracking_number":"TRACK-1","customer_email":"private@example.com"}'
        )
        urlopen.return_value = response

        result = lookup_order("ABCD-1234")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://orders.example/api/orders/ABCD-1234")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + "test-token")
        self.assertEqual(result, "Status: shipped\nTracking number: TRACK-1")

    def test_refund_pauses_until_a_human_decides(self):
        graph = create_support_graph(llm=FakeModel(), documents=self.documents)
        config = {"configurable": {"thread_id": "refund-test"}}
        paused = graph.invoke(
            {"messages": [HumanMessage(content="I want a refund")]}, config=config
        )

        self.assertEqual(paused["__interrupt__"][0].value["type"], "refund_approval")
        approved = graph.invoke(Command(resume={"approved": True}), config=config)
        self.assertIn("No refund was issued automatically", approved["response"])

    def test_rejected_refund_is_not_processed(self):
        graph = create_support_graph(llm=FakeModel(), documents=self.documents)
        config = {"configurable": {"thread_id": "refund-rejected-test"}}
        graph.invoke(
            {"messages": [HumanMessage(content="Please reimburse my purchase")]},
            config=config,
        )
        rejected = graph.invoke(Command(resume={"approved": False}), config=config)

        self.assertIn("No refund was issued", rejected["response"])


if __name__ == "__main__":
    unittest.main()
