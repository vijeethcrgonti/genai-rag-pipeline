"""
rag_stack.py  —  cdk/lib/
Provisions OpenSearch Serverless, DynamoDB tables, Lambda (containerized),
API Gateway, and IAM roles for the RAG pipeline.
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_opensearchserverless as oss,
)
from constructs import Construct


class RAGStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, stage: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ── DynamoDB: Conversation Memory ──────────────────────────────────────

        memory_table = dynamodb.Table(
            self,
            "ConversationMemory",
            table_name=f"rag-conversation-memory-{stage}",
            partition_key=dynamodb.Attribute(
                name="session_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            time_to_live_attribute="ttl",
        )

        # ── DynamoDB: Embedding Cache ──────────────────────────────────────────

        embedding_cache = dynamodb.Table(
            self,
            "EmbeddingCache",
            table_name=f"rag-embedding-cache-{stage}",
            partition_key=dynamodb.Attribute(
                name="content_hash", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            time_to_live_attribute="ttl",
        )

        # ── DynamoDB: Chunk Dedup ──────────────────────────────────────────────

        dedup_table = dynamodb.Table(
            self,
            "ChunkDedup",
            table_name=f"rag-chunk-dedup-{stage}",
            partition_key=dynamodb.Attribute(
                name="content_hash", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # ── OpenSearch Serverless Collection ───────────────────────────────────

        collection = oss.CfnCollection(
            self,
            "KnowledgeBaseCollection",
            name=f"rag-kb-{stage}",
            type="VECTORSEARCH",
            description="RAG pipeline knowledge base vector index",
        )

        # ── Lambda IAM Role ────────────────────────────────────────────────────

        lambda_role = iam.Role(
            self,
            "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["arn:aws:bedrock:*::foundation-model/*"],
            )
        )

        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[collection.attr_arn],
            )
        )

        for table in [memory_table, embedding_cache, dedup_table]:
            table.grant_read_write_data(lambda_role)

        # ── Lambda Function (containerized) ───────────────────────────────────

        api_fn = lambda_.DockerImageFunction(
            self,
            "RAGApiFunction",
            function_name=f"rag-api-{stage}",
            code=lambda_.DockerImageCode.from_image_asset(".."),
            role=lambda_role,
            timeout=cdk.Duration.seconds(60),
            memory_size=1024,
            environment={
                "OPENSEARCH_ENDPOINT": collection.attr_collection_endpoint,
                "OPENSEARCH_INDEX": "rag-knowledge-base",
                "MEMORY_TABLE": memory_table.table_name,
                "EMBEDDING_CACHE_TABLE": embedding_cache.table_name,
                "DEDUP_TABLE": dedup_table.table_name,
                "AWS_REGION": self.region,
                "STAGE": stage,
            },
        )

        # ── API Gateway ────────────────────────────────────────────────────────

        api = apigw.RestApi(
            self,
            "RAGAPI",
            rest_api_name=f"rag-api-{stage}",
            description="GenAI RAG Pipeline API",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        lambda_integration = apigw.LambdaIntegration(api_fn)

        query = api.root.add_resource("query")
        query.add_method("POST", lambda_integration)

        ingest = api.root.add_resource("ingest")
        ingest.add_method("POST", lambda_integration)

        health = api.root.add_resource("health")
        health.add_method("GET", lambda_integration)

        metrics = api.root.add_resource("metrics")
        metrics.add_method("GET", lambda_integration)

        # ── Outputs ────────────────────────────────────────────────────────────

        cdk.CfnOutput(self, "APIEndpoint", value=api.url)
        cdk.CfnOutput(
            self, "OpenSearchEndpoint", value=collection.attr_collection_endpoint
        )
        cdk.CfnOutput(self, "LambdaFunctionName", value=api_fn.function_name)
