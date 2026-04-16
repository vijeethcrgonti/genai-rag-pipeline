import aws_cdk as cdk
from lib.rag_stack import RAGStack

app = cdk.App()
stage = app.node.try_get_context("stage") or "dev"
RAGStack(app, "RAGStack", stage=stage)
app.synth()
