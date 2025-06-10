# SQLAlchemy + PostgreSQL 155 Querysets Reference

This document provides a complete reference for 155 SQLAlchemy querysets optimized for PostgreSQL, covering basic CRUD, filtering, joins, PostgreSQL-specific features (JSONB, arrays, full-text search), advanced analytics, AI-driven vector search, real-time systems, distributed queries, and autonomous systems. Each query includes a concise example using SQLAlchemy 2.x. For full code, test data, and setup scripts, visit [https://github.com/xai/sqlalchemy-querysets](https://github.com/xai/sqlalchemy-querysets).

## Table of Contents
- [Setup](#setup)
- [Basic CRUD (Queries 1–6)](#basic-crud-queries-1-6)
- [Filtering and Sorting (Queries 7–20)](#filtering-and-sorting-queries-7-20)
- [Pagination (Queries 21–22)](#pagination-queries-21-22)
- [Aggregations (Queries 23–28)](#aggregations-queries-23-28)
- [Joins and Relationships (Queries 29–34)](#joins-and-relationships-queries-29-34)
- [PostgreSQL-Specific Features (Queries 35–50)](#postgresql-specific-features-queries-35-50)
- [Advanced Analytics (Queries 51–70)](#advanced-analytics-queries-51-70)
- [Bulk Operations and Upserts (Queries 71–75)](#bulk-operations-and-upserts-queries-71-75)
- [Real-Time Systems (Queries 76–80)](#real-time-systems-queries-76-80)
- [AI and Vector Search (Queries 81–90)](#ai-and-vector-search-queries-81-90)
- [Monitoring and Metrics (Queries 91–95)](#monitoring-and-metrics-queries-91-95)
- [Distributed Queries (Queries 96–100)](#distributed-queries-queries-96-100)
- [Multi-Tenant and Security (Queries 101–105)](#multi-tenant-and-security-queries-101-105)
- [Advanced Real-Time (Queries 106–115)](#advanced-real-time-queries-106-115)
- [AI-Driven Systems (Queries 116–135)](#ai-driven-systems-queries-116-135)
- [Autonomous Systems (Queries 136–155)](#autonomous-systems-queries-136-155)
- [Conclusion](#conclusion)

## Setup
```python
from sqlalchemy import create_engine, select, update, delete, func, and_, or_, desc, case, text
from sqlalchemy.orm import sessionmaker, joinedload, aliased
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, TSRANGE, Ltree
from pgvector.sqlalchemy import Vector
from my_app.models import User, Department, Post, Category, Document, Event, Order, Review, Metric
from datetime import datetime, timedelta

engine = create_engine("postgresql+psycopg2://user:pass@localhost:5432/db")
Session = sessionmaker(bind=engine)
session = Session()
```

## Basic CRUD (Queries 1–6)
1. **Fetch All Records**: Retrieve all users.
   ```python
   users = session.execute(select(User)).scalars().all()
   ```
2. **Fetch First Record**: Get the first user.
   ```python
   user = session.execute(select(User)).scalars().first()
   ```
3. **Fetch by ID**: Retrieve a user by ID.
   ```python
   user = session.get(User, 1)
   ```
4. **Create Record**: Insert a new user.
   ```python
   new_user = User(name="Alice", email="alice@example.com", status="active", age=30)
   session.add(new_user)
   session.commit()
   ```
5. **Update Record**: Update user fields.
   ```python
   session.execute(update(User).where(User.id == 1).values(name="Updated Alice"))
   session.commit()
   ```
6. **Delete Record**: Delete a user by ID.
   ```python
   session.execute(delete(User).where(User.id == 1))
   session.commit()
   ```

## Filtering and Sorting (Queries 7–20)
7. **Filter by Equality**: Find active users.
   ```python
   users = session.execute(select(User).where(User.status == "active")).scalars().all()
   ```
8. **Multiple Conditions**: Filter by status and age.
   ```python
   users = session.execute(select(User).where(and_(User.status == "active", User.age > 18))).scalars().all()
   ```
9. **Filter with OR**: Search by name or email.
   ```python
   users = session.execute(select(User).where(or_(User.name.ilike("%alice%"), User.email.ilike("%example.com")))).scalars().all()
   ```
10. **Filter with IN**: Find users in countries.
    ```python
    users = session.execute(select(User).where(User.country.in_(["USA", "India"]))).scalars().all()
    ```
11. **Case-Insensitive LIKE**: Search names.
    ```python
    users = session.execute(select(User).where(User.name.ilike("%alice%"))).scalars().all()
    ```
12. **Order Ascending**: Sort by age.
    ```python
    users = session.execute(select(User).order_by(User.age)).scalars().all()
    ```
13. **Order Descending**: Sort by age descending.
    ```python
    users = session.execute(select(User).order_by(desc(User.age))).scalars().all()
    ```
14. **Filter by NULL**: Find users with no email.
    ```python
    users = session.execute(select(User).where(User.email.is_(None))).scalars().all()
    ```
15. **Filter by NOT NULL**: Find users with email.
    ```python
    users = session.execute(select(User).where(User.email.isnot(None))).scalars().all()
    ```
16. **Date Range**: Filter by creation date.
    ```python
    start = datetime.now() - timedelta(days=30)
    users = session.execute(select(User).where(User.created_at.between(start, datetime.now()))).scalars().all()
    ```
17. **Exists**: Check if user exists.
    ```python
    from sqlalchemy.sql import exists
    has_user = session.execute(select(exists().where(User.email == "alice@example.com"))).scalar()
    ```
18. **Regex Filter**: Find Gmail users.
    ```python
    users = session.execute(select(User).where(User.email.op("~*")(".*@gmail.com$"))).scalars().all()
    ```
19. **Month Extraction**: Filter by creation month.
    ```python
    users = session.execute(select(User).where(func.extract("month", User.created_at) == 6)).scalars().all()
    ```
20. **Dynamic Filtering**: Apply dynamic filters.
    ```python
    filters = {"status": "active"}
    query = select(User)
    for k, v in filters.items():
        query = query.where(getattr(User, k) == v)
    users = session.execute(query).scalars().all()
    ```

## Pagination (Queries 21–22)
21. **Limit**: Fetch first 10 users.
    ```python
    users = session.execute(select(User).limit(10)).scalars().all()
    ```
22. **Offset and Limit**: Paginate users.
    ```python
    page, per_page = 2, 10
    users = session.execute(select(User).offset((page - 1) * per_page).limit(per_page)).scalars().all()
    ```

## Aggregations (Queries 23–28)
23. **Count Total**: Count all users.
    ```python
    count = session.execute(select(func.count(User.id))).scalar()
    ```
24. **Group By**: Count users by country.
    ```python
    results = session.execute(select(User.country, func.count(User.id)).group_by(User.country)).all()
    ```
25. **Conditional Count**: Count active users.
    ```python
    counts = session.execute(select(func.count().filter(User.status == "active").label("active"))).one()
    ```
26. **Having Clause**: Countries with >10 users.
    ```python
    results = session.execute(select(User.country, func.count(User.id)).group_by(User.country).having(func.count(User.id) > 10)).all()
    ```
27. **Average**: Average user age.
    ```python
    avg_age = session.execute(select(func.avg(User.age))).scalar()
    ```
28. **Max**: Latest user creation date.
    ```python
    latest = session.execute(select(func.max(User.created_at))).scalar()
    ```

## Joins and Relationships (Queries 29–34)
29. **Inner Join**: Join User with Department.
    ```python
    results = session.execute(select(User, Department).join(Department, User.department_id == Department.id)).all()
    ```
30. **Eager Loading**: Load User with Department.
    ```python
    users = session.execute(select(User).options(joinedload(User.department))).scalars().all()
    ```
31. **Self Join**: Join User with Manager.
    ```python
    Manager = aliased(User)
    results = session.execute(select(User.name, Manager.name.label("manager_name")).join(Manager, User.manager_id == Manager.id)).all()
    ```
32. **Outer Join**: Left join User with Department.
    ```python
    results = session.execute(select(User, Department).outerjoin(Department, User.department_id == Department.id)).all()
    ```
33. **Multiple Joins**: Join User, Order, Review.
    ```python
    results = session.execute(select(User).join(Order).join(Review)).all()
    ```
34. **Lateral Join**: Latest event per user.
    ```python
    latest_log = select(Event).where(Event.user_id == User.id).order_by(desc(Event.created_at)).limit(1).lateral()
    results = session.execute(select(User.name, latest_log.c.created_at).join(latest_log, true())).all()
    ```

## PostgreSQL-Specific Features (Queries 35–50)
35. **JSONB Key Filter**: Filter by JSONB city.
    ```python
    users = session.execute(select(User).where(User.profile["city"].astext == "Delhi")).scalars().all()
    ```
36. **JSONB Path Filter**: Filter by nested JSONB.
    ```python
    users = session.execute(select(User).where(User.profile["address"]["city"].astext == "Delhi")).scalars().all()
    ```
37. **JSONB Key Exists**: Check for JSONB key.
    ```python
    users = session.execute(select(User).where(User.profile.has_key("age"))).scalars().all()
    ```
38. **Array Contains**: Find users with skill.
    ```python
    users = session.execute(select(User).where(User.skills.contains(["python"]))).scalars().all()
    ```
39. **Array Any**: Filter by any skill.
    ```python
    users = session.execute(select(User).where("python" == any_(User.skills))).scalars().all()
    ```
40. **Full-Text Search**: Search bios for "developer".
    ```python
    results = session.execute(select(User).where(func.to_tsvector("english", User.bio).match("developer"))).scalars().all()
    ```
41. **Range Overlap**: Filter by active period.
    ```python
    users = session.execute(select(User).where(User.active_period.op("&&")("[2024-01-01,2024-12-31]"))).scalars().all()
    ```
42. **Ltree Path**: Query hierarchical categories.
    ```python
    categories = session.execute(select(Category).where(Category.path.descendant_of("root.A.B"))).scalars().all()
    ```
43. **JSON Aggregation**: Aggregate user JSON.
    ```python
    results = session.execute(select(Department.name, func.json_agg(func.json_build_object("id", User.id, "name", User.name))).join(User).group_by(Department.name)).all()
    ```
44. **Generate Series**: Create date series.
    ```python
    results = session.execute(select(func.generate_series("2024-01-01", "2024-12-31", "1 month"))).all()
    ```
45. **Array Unnest**: Expand tags array.
    ```python
    results = session.execute(text("SELECT u.id, tag FROM users u, LATERAL unnest(u.tags) AS tag WHERE tag = :tag"), {"tag": "python"}).all()
    ```
46. **Time Truncation**: Group by month.
    ```python
    results = session.execute(select(func.date_trunc("month", Event.created_at), func.count(Event.id)).group_by(func.date_trunc("month", Event.created_at))).all()
    ```
47. **Enum Filtering**: Filter by status enum.
    ```python
    users = session.execute(select(User).where(User.status == "active")).scalars().all()
    ```
48. **PostGIS Distance**: Find users within 5km.
    ```python
    from geoalchemy2 import Geography
    results = session.execute(select(User).where(func.ST_DWithin(User.location, "POINT(77.5946 12.9716)", 5000))).scalars().all()
    ```
49. **Tablesample**: Random sample of users.
    ```python
    results = session.execute(text("SELECT * FROM users TABLESAMPLE SYSTEM(10)")).all()
    ```
50. **Distinct**: Unique user names.
    ```python
    names = session.execute(select(User.name).distinct()).scalars().all()
    ```

## Advanced Analytics (Queries 51–70)
51. **Window Row Number**: Rank posts per user.
    ```python
    stmt = select(Post, func.row_number().over(partition_by=Post.user_id, order_by=desc(Post.created_at)).label("rn")).subquery()
    results = session.execute(select(stmt).where(stmt.c.rn <= 2)).all()
    ```
52. **Cumulative Sum**: Running total of orders.
    ```python
    results = session.execute(select(Order.id, func.sum(Order.total).over(partition_by=Order.user_id, order_by=Order.date).label("total"))).all()
    ```
53. **Recursive CTE**: Hierarchical categories.
    ```python
    cte = select(Category).where(Category.parent_id.is_(None)).cte(name="tree", recursive=True)
    cte = cte.union_all(select(Category).join(cte, Category.parent_id == cte.c.id))
    results = session.execute(select(cte)).all()
    ```
54. **CASE Expression**: Classify age groups.
    ```python
    results = session.execute(select(User.name, case((User.age < 18, "Minor"), (User.age >= 18, "Adult")).label("age_group"))).all()
    ```
55. **Subquery Filter**: Filter by department.
    ```python
    subq = select(Department.id).where(Department.name == "Engineering").subquery()
    users = session.execute(select(User).where(User.department_id.in_(subq))).scalars().all()
    ```
56. **Union**: Combine queries.
    ```python
    q1 = select(User.name).where(User.country == "India")
    q2 = select(User.name).where(User.country == "USA")
    results = session.execute(q1.union(q2)).scalars().all()
    ```
57. **Correlated Subquery**: Latest post per user.
    ```python
    latest_post = select(Post).where(Post.user_id == User.id).order_by(desc(Post.created_at)).limit(1).correlate(User).as_scalar()
    results = session.execute(select(User.name, latest_post.label("latest_post_date"))).all()
    ```
58. **RFM Analysis**: Recency, frequency, monetary.
    ```python
    results = session.execute(select(User.id, func.max(Order.date).label("last_purchase"), func.count(Order.id).label("frequency"), func.sum(Order.total).label("monetary")).join(Order).group_by(User.id)).all()
    ```
59. **Multi-Column Group By**: Group by country, status.
    ```python
    results = session.execute(select(User.country, User.status, func.count(User.id)).group_by(User.country, User.status)).all()
    ```
60. **Time Since**: Days since last login.
    ```python
    results = session.execute(select(User.name, (func.now() - User.last_login).label("days_since_login"))).all()
    ```
61. **String Aggregation**: Aggregate user names.
    ```python
    results = session.execute(select(Department.name, func.string_agg(User.name, ", ").label("users")).join(User).group_by(Department.name)).all()
    ```
62. **Latest Per Group**: Latest post per user.
    ```python
    subq = select(Post.user_id, func.max(Post.created_at).label("latest")).group_by(Post.user_id).subquery()
    results = session.execute(select(Post).join(subq, and_(Post.user_id == subq.c.user_id, Post.created_at == subq.c.latest))).all()
    ```
63. **Dynamic Metrics**: Count and average age.
    ```python
    metrics = session.execute(select(func.count().label("count"), func.avg(User.age).label("avg_age")).where(User.status == "active")).one()
    ```
64. **Nested JSON**: Aggregate nested JSON.
    ```python
    results = session.execute(select(func.json_build_object("dept", Department.name, "users", func.json_agg(func.json_build_object("id", User.id, "name", User.name)))).join(User).group_by(Department.name)).all()
    ```
65. **Annotated Counts**: Annotate post counts.
    ```python
    subq = select(Post.user_id, func.count(Post.id).label("count")).group_by(Post.user_id).subquery()
    results = session.execute(select(User, subq.c.count).outerjoin(subq, User.id == subq.c.user_id)).all()
    ```
66. **Composite Key Filter**: Filter by user, category.
    ```python
    from sqlalchemy.sql import tuple_
    results = session.execute(select(Post).where(tuple_(Post.user_id, Post.category_id).in_([(1, 2), (3, 4)]))).scalars().all()
    ```
67. **Polymorphic Query**: Query vehicle subclasses.
    ```python
    results = session.execute(select(Vehicle).with_polymorphic("*")).scalars().all()
    ```
68. **Multi-Field Search**: Search name or email.
    ```python
    results = session.execute(select(User).where(or_(User.name.ilike("%term%"), User.email.ilike("%term%")))).scalars().all()
    ```
69. **Explain Plan**: Analyze query performance.
    ```python
    results = session.execute(text("EXPLAIN SELECT * FROM users WHERE name ILIKE '%alice%'")).all()
    ```
70. **BETWEEN Condition**: Filter by date range.
    ```python
    results = session.execute(select(User).where(User.created_at.between(start_date, end_date))).scalars().all()
    ```

## Bulk Operations and Upserts (Queries 71–75)
71. **Bulk Insert**: Insert multiple users.
    ```python
    session.add_all([User(name=f"User{i}") for i in range(100)])
    session.commit()
    ```
72. **Bulk Update**: Update multiple users.
    ```python
    session.bulk_update_mappings(User, [{"id": 1, "name": "Updated"}, {"id": 2, "name": "Changed"}])
    session.commit()
    ```
73. **Upsert**: Insert or update user.
    ```python
    from sqlalchemy.dialects.postgresql import insert
    stmt = insert(User).values(id=1, name="Alice").on_conflict_do_update(index_elements=["id"], set_={"name": "Updated Alice"})
    session.execute(stmt)
    session.commit()
    ```
74. **Insert with RETURNING**: Return inserted ID.
    ```python
    user_id = session.execute(insert(User).values(name="New User").returning(User.id)).scalar()
    ```
75. **Insert or Do Nothing**: Ignore conflicts.
    ```python
    stmt = insert(User).values(id=1, name="Alice").on_conflict_do_nothing(index_elements=["id"])
    session.execute(stmt)
    session.commit()
    ```

## Real-Time Systems (Queries 76–80)
76. **LISTEN/NOTIFY**: Listen for updates.
    ```python
    import asyncpg
    import asyncio
    async def listen(dsn):
        conn = await asyncpg.connect(dsn)
        await conn.add_listener("data_changes", lambda *args: print("Update:", args[3]))
        while True:
            await asyncio.sleep(1)
    ```
77. **Streaming Results**: Process large datasets.
    ```python
    for user in session.query(User).yield_per(1000):
        process(user)
    ```
78. **Row Locking**: Lock queued jobs.
    ```python
    results = session.execute(select(Job).with_for_update(skip_locked=True).where(Job.status == "queued").limit(5)).scalars().all()
    ```
79. **Trigger Event**: Set created_at on insert.
    ```python
    from sqlalchemy import event
    @event.listens_for(User, "before_insert")
    def set_created_at(mapper, connection, target):
        target.created_at = datetime.utcnow()
    ```
80. **Materialized View Refresh**: Refresh view.
    ```python
    session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY top_users_mv"))
    ```

## AI and Vector Search (Queries 81–90)
81. **Vector Search**: Find similar documents.
    ```python
    docs = session.execute(select(Document).order_by(Document.embedding.l2_distance(query_vector)).limit(5)).scalars().all()
    ```
82. **RAG Pipeline**: LLM-driven query.
    ```python
    from langchain.sql_database import SQLDatabase
    from langchain.chat_models import ChatOpenAI
    db = SQLDatabase.from_uri("postgresql+psycopg2://user:pass@localhost/db")
    llm = ChatOpenAI(temperature=0)
    sql_chain = SQLDatabaseChain.from_llm(llm, db)
    result = sql_chain.run("List users with more than 5 orders in 2023")
    ```
83. **Embedding Rebuild**: Generate embeddings.
    ```python
    docs = session.execute(select(Document).where(Document.embedding == None)).scalars().all()
    for doc in docs:
        doc.embedding = model.embed(doc.content)
    session.commit()
    ```
84. **Hybrid Search**: Combine vector and category.
    ```python
    docs = session.execute(select(Document).where(Document.category == "news").order_by(Document.embedding.l2_distance(query_vector)).limit(5)).scalars().all()
    ```
85. **Semantic + Full-Text**: Combine search types.
    ```python
    docs = session.execute(select(Document).where(Document.tags.op("@@")(func.to_tsquery("chatbot"))).order_by(Document.embedding.l2_distance(query_vector)).limit(5)).scalars().all()
    ```
86. **Embedding Drift**: Detect drift in embeddings.
    ```python
    results = session.execute(select(EmbeddingDrift).where(EmbeddingDrift.drift_score > 0.7).order_by(desc(EmbeddingDrift.timestamp)).limit(20)).scalars().all()
    ```
87. **AI Query Mapping**: Convert SQL to ORM.
    ```python
    prompt = "Map this SQL to ORM: SELECT name FROM users WHERE age > 20"
    query = llm.run(prompt)
    ```
88. **Context-Aware Search**: Use query history.
    ```python
    history = session.execute(select(QueryLog).where(QueryLog.user_id == user_id).order_by(desc(QueryLog.timestamp)).limit(5)).scalars().all()
    context = " | ".join([h.query for h in history])
    result = sql_chain.run(f"Based on {context}, answer: {query}")
    ```
89. **Prediction Table Query**: Churn risk analysis.
    ```python
    results = session.execute(select(UserPrediction).where(UserPrediction.churn_risk > 0.8)).scalars().all()
    ```
90. **Multi-Table Dataset**: User order stats.
    ```python
    results = session.execute(select(User.id, User.country, func.count(Order.id).label("orders"), func.sum(Order.total).label("spent")).join(Order).group_by(User.id, User.country)).all()
    ```

## Monitoring and Metrics (Queries 91–95)
91. **Prometheus Metrics**: Export active users.
    ```python
    from prometheus_client import Gauge, start_http_server
    active_users = Gauge("active_users", "Count of active users")
    start_http_server(8000)
    while True:
        count = session.execute(select(func.count(User.id)).where(User.status == "active")).scalar()
        active_users.set(count)
        time.sleep(60)
    ```
92. **KPI Alert**: Alert on high response time.
    ```python
    avg_time = session.execute(select(func.avg(Metric.value)).where(Metric.name == "response_time")).scalar()
    if avg_time > 5.0:
        alert(f"High response time: {avg_time:.2f} sec")
    ```
93. **Model Drift**: Compare model versions.
    ```python
    prod = session.execute(select(Metric).where(Metric.version == "prod")).scalars().all()
    test = session.execute(select(Metric).where(Metric.version == "new")).scalars().all()
    diff = [abs(p.value - t.value) for p, t in zip(prod, test)]
    ```
94. **Audit Log Summary**: Summarize audit logs.
    ```python
    results = session.execute(select(AuditLog.model_name, func.count().label("hits")).group_by(AuditLog.model_name)).all()
    ```
95. **Query Log Analysis**: Analyze failed queries.
    ```python
    results = session.execute(select(QueryLog.query, QueryLog.latency).where(QueryLog.success == False)).all()
    ```

## Distributed Queries (Queries 96–100)
96. **Trino Query**: Query S3 via Trino.
    ```python
    from trino.sqlalchemy import URL
    trino_engine = create_engine(URL(host="trino-host", port=8080, catalog="s3", schema="default"))
    results = trino_engine.execute("SELECT * FROM s3.default.events").fetchall()
    ```
97. **Federated Query**: Query remote schema.
    ```python
    results = session.execute(text("SELECT * FROM foreign_schema.remote_users")).all()
    ```
98. **Multi-Schema Query**: Query tenant schema.
    ```python
    schema = "tenant_1_schema"
    results = session.execute(text(f"SELECT * FROM {schema}.users")).all()
    ```
99. **Shard Query**: Query specific shard.
    ```python
    def get_shard_session(shard_id):
        return Session(bind=engines[shard_id])
    results = get_shard_session("shard_1").execute(select(User)).scalars().all()
    ```
100. **External SQL File**: Execute SQL file.
     ```python
     with open("query.sql") as f:
         results = session.execute(text(f.read()), {"year": 2024}).all()
     ```

## Multi-Tenant and Security (Queries 101–105)
101. **Tenant Isolation**: Filter by tenant ID.
     ```python
     results = session.execute(select(Project).where(Project.tenant_id == user.tenant_id)).scalars().all()
     ```
102. **Role-Based Filter**: Restrict by user role.
     ```python
     query = select(User)
     if not user.is_superuser:
         query = query.where(User.owner_id == user.id)
     results = session.execute(query).scalars().all()
     ```
103. **RLS Filter**: Row-level security.
     ```python
     results = session.execute(select(Resource).where(or_(Resource.owner_id == user.id, Resource.group_id.in_(user.group_ids)))).scalars().all()
     ```
104. **Organization Hierarchy**: Filter by org.
     ```python
     results = session.execute(select(Task).where(Task.organization_id == user.organization_id, Task.project_id.in_(user.project_ids))).scalars().all()
     ```
105. **Access Log**: User access history.
     ```python
     results = session.execute(select(AuditLog).where(AuditLog.user_id == user.id)).scalars().all()
     ```

## Advanced Real-Time (Queries 106–115)
106. **Real-Time Dashboard**: Notify on insert.
     ```python
     @event.listens_for(User, "after_insert")
     def notify(mapper, conn, target):
         conn.execute(text("NOTIFY data_changes, 'user_inserted'"))
     ```
107. **Logical Replication**: Track transactions.
     ```python
     results = session.execute(text("SELECT pg_xact_commit_timestamp(xmin) FROM users")).all()
     ```
108. **Time Travel**: Historical user data.
     ```python
     results = session.execute(select(UserHistory).where(UserHistory.user_id == user_id, UserHistory.timestamp <= target_time).order_by(desc(UserHistory.timestamp)).first()).scalars().all()
     ```
109. **Event Sourcing**: Log entity events.
     ```python
     results = session.execute(select(EventLog).where(EventLog.entity_type == "User").order_by(EventLog.timestamp)).scalars().all()
     ```
110. **WebSocket Notify**: Real-time updates.
     ```python
     async def notify():
         conn = await asyncpg.connect(dsn)
         await conn.add_listener("orders", lambda *args: print("Notify:", args[3]))
         while True:
             await asyncio.sleep(1)
     ```
111. **Materialized View**: Query view.
     ```python
     class TopUser(Base):
         __tablename__ = "top_users_mv"
         __table_args__ = {'autoload_with': engine}
     results = session.execute(select(TopUser)).scalars().all()
     ```
112. **Partition Query**: Query event partition.
     ```python
     results = session.execute(select(Event).where(Event.ts.between(start, end))).scalars().all()
     ```
113. **Timescale Bucket**: Bucket metrics.
     ```python
     results = session.execute(select(func.time_bucket("1 hour", Metric.timestamp), func.avg(Metric.cpu)).group_by(func.time_bucket("1 hour", Metric.timestamp))).all()
     ```
114. **Live Status**: Task status counts.
     ```python
     results = session.execute(text("SELECT status, COUNT(*) FROM tasks GROUP BY status")).all()
     ```
115. **Query Factory**: Generic query builder.
     ```python
     class QueryFactory:
         def all(self, model):
             return session.execute(select(model)).scalars().all()
     factory = QueryFactory()
     results = factory.all(User)
     ```

## AI-Driven Systems (Queries 116–135)
116. **SQLChat Agent**: Natural language queries.
     ```python
     from langchain.chains import SQLDatabaseChain
     db = SQLDatabase.from_uri("postgresql+psycopg2://user:pass@localhost/db")
     llm = ChatOpenAI(temperature=0)
     agent = SQLDatabaseChain.from_llm(llm, db)
     result = agent.run("List projects started after 2023")
     ```
117. **Query Finetuning**: Log query performance.
     ```python
     session.execute(text("INSERT INTO query_logs (query, latency, success) VALUES (:q, :l, :s)"), {"q": "SELECT *", "l": 0.1, "s": 1})
     ```
118. **Hybrid Memory**: Query memory logs.
     ```python
     results = session.execute(select(MemoryLog).order_by(MemoryLog.embedding.l2_distance(prompt_vector)).limit(5)).scalars().all()
     ```
119. **LLM Validation**: Validate LLM query.
     ```python
     try:
         results = session.execute(text(llm.run("SQL for: List users")))
     except Exception as e:
         print(f"Query failed: {e}")
     ```
120. **Top-K Vector**: Top 10 similar documents.
     ```python
     docs = session.execute(select(Document).where(Document.category == "legal").order_by(Document.embedding.l2_distance(query_vector)).limit(10)).scalars().all()
     ```
121. **Secure Vector**: Tenant-specific vector search.
     ```python
     docs = session.execute(select(Document).where(Document.user_id == user.id).order_by(Document.embedding.l2_distance(query_vector)).limit(10)).scalars().all()
     ```
122. **LLM Cache**: Cache LLM responses.
     ```python
     entry = session.execute(select(LlmCache).where(LlmCache.prompt == prompt)).scalars().first()
     ```
123. **Graph Traversal**: Traverse relationships.
     ```python
     results = session.execute(select(Category).join(Post).join(Order).join(User).where(User.email == "a@b.com").distinct()).scalars().all()
     ```
124. **Time-Decay Ranking**: Rank posts by recency.
     ```python
     results = session.execute(select(Post).order_by((func.extract("epoch", func.now() - Post.created_at) / 3600).asc()).limit(10)).scalars().all()
     ```
125. **Agent Memory**: Query agent memory.
     ```python
     results = session.execute(select(AIAgentMemory).where(AIAgentMemory.agent == "support_bot").order_by(desc(AIAgentMemory.timestamp)).limit(20)).scalars().all()
     ```
126. **Anomaly Detection**: Detect transaction anomalies.
     ```python
     import pandas as pd
     from sklearn.ensemble import IsolationForest
     df = pd.read_sql(select(Transaction).statement, session.bind)
     model = IsolationForest().fit(df[["amount"]])
     df["anomaly"] = model.predict(df[["amount"]])
     ```
127. **Delta Query**: Fetch updated records.
     ```python
     results = session.execute(select(User).where(User.updated_at > last_sync_time)).scalars().all()
     ```
128. **Data Versioning**: Latest user version.
     ```python
     results = session.execute(select(UserHistory).where(UserHistory.user_id == uid).order_by(desc(UserHistory.timestamp)).limit(1)).scalars().all()
     ```
129. **Index Hint**: Force index usage.
     ```python
     session.execute(text("SET enable_seqscan = OFF; SELECT * FROM users WHERE email LIKE '%gmail.com'")).all()
     ```
130. **AI Query Builder**: Build query from NL.
     ```python
     class AIQueryBuilder:
         def run(self, nl_query):
             sql = llm.run(f"Write PostgreSQL query for: {nl_query}")
             return session.execute(text(sql)).all()
     ```
131. **Temporal Join**: Join with history.
     ```python
     results = session.execute(select(User, UserHistory).where(User.id == UserHistory.user_id, UserHistory.timestamp < User.updated_at)).all()
     ```
132. **Multi-Store Fusion**: Combine Neo4j and Postgres.
     ```python
     from neo4j import GraphDatabase
     with neo4j_driver.session() as neo:
         graph = neo.run("MATCH (u:User)-[:FOLLOWS]->(f) RETURN f.id").data()
     docs = session.execute(select(Document).order_by(Document.embedding.l2_distance(query_vector)).limit(5)).scalars().all()
     ```
133. **Ephemeral Cache**: Query uncached documents.
     ```python
     results = session.execute(select(Document).where(Document.embedding == None)).scalars().all()
     ```
134. **Auto-Gated Query**: Restrict by role.
     ```python
     def gated_query(model, user):
         query = select(model)
         if not user.is_superuser:
             query = query.where(model.org_id == user.org_id)
         return session.execute(query).scalars().all()
     ```
135. **Smart Pagination**: Dynamic pagination.
     ```python
     def paged_query(model, search=None, sort="-created", page=1, limit=20):
         query = select(model)
         if search:
             query = query.where(model.name.ilike(f"%{search}%"))
         attr = getattr(model, sort.lstrip("-"))
         query = query.order_by(desc(attr) if sort.startswith("-") else attr)
         return session.execute(query.offset((page - 1) * limit).limit(limit)).scalars().all()
     ```

## Autonomous Systems (Queries 136–155)
136. **Self-Updating Dashboard**: Notify on insert.
     ```python
     @event.listens_for(User, "after_insert")
     def notify(mapper, conn, target):
         conn.execute(text("NOTIFY data_changes, 'user_inserted'"))
     ```
137. **Real-Time Pipeline**: Stream updates.
     ```python
     async def stream():
         conn = await asyncpg.connect(dsn)
         await conn.add_listener("orders", lambda *args: update_analytics(args[3]))
         while True:
             await asyncio.sleep(1)
     ```
138. **AI Agent Query**: Run NL query.
     ```python
     prompt = "Show last 5 projects"
     query = langchain.run(prompt)
     results = session.execute(text(query)).all()
     ```
139. **Geospatial Query**: Find nearby locations.
     ```python
     results = session.execute(select(Location).where(func.ST_DWithin(Location.point, func.ST_MakePoint(lon, lat), 10000))).scalars().all()
     ```
140. **Live KPI Alert**: Alert on KPIs.
     ```python
     result = session.execute(select(func.avg(Metric.value)).where(Metric.name == "response_time")).scalar()
     if result > 5.0:
         slack.send_alert(f"Response time high: {result:.2f} sec")
     ```
141. **PGVector Rebuild**: Rebuild embeddings.
     ```python
     docs = session.execute(select(Document).where(Document.embedding == None)).scalars().all()
     for doc in docs:
         doc.embedding = model.embed(doc.content)
     session.commit()
     ```
142. **Self-Optimizing RAG**: RAG pipeline.
     ```python
     docs = session.execute(select(Document).order_by(Document.embedding.l2_distance(vector)).limit(5)).scalars().all()
     context = "\n".join([doc.content for doc in docs])
     response = llm_chain.run(f"Based on {context}, answer: {query}")
     ```
143. **Distributed Query Routing**: Route queries.
     ```python
     class QueryRouter:
         def route(self, query):
             if "analytics" in query.lower():
                 return duckdb_session.execute(query)
             return session.execute(text(query)).all()
     ```
144. **Prometheus Exporter**: Export metrics.
     ```python
     active_users = Gauge("active_users", "Count of active users")
     start_http_server(8000)
     while True:
         count = session.execute(select(func.count(User.id)).where(User.status == "active")).scalar()
         active_users.set(count)
         time.sleep(60)
     ```
145. **IDE Plugin Template**: Generate ORM queries.
     ```python
     class SQLPlugin:
         def generate_orm(self, prompt):
             query = llm.run(f"Convert to SQLAlchemy ORM: {prompt}")
             if "exec(" in query:
                 raise ValueError("SQL injection risk")
             return query
     ```
146. **Query Benchmarking**: Measure latency.
     ```python
     start = time.time()
     results = session.execute(select(User)).scalars().all()
     latency = time.time() - start
     ```
147. **Auto Schema Optimization**: Analyze tables.
     ```python
     results = session.execute(text("ANALYZE users")).all()
     ```
148. **Cross-Engine Orchestration**: Union across engines.
     ```python
     results = trino_engine.execute("SELECT * FROM postgres.public.users UNION SELECT * FROM s3.default.events").fetchall()
     ```
149. **Real-Time RAG Agent**: Run RAG agent.
     ```python
     agent = RAGAgent(session, llm)
     results = agent.run("Find recent projects")
     ```
150. **WebSocket Dashboard**: Stream updates.
     ```python
     async def dashboard():
         conn = await asyncpg.connect(dsn)
         await conn.add_listener("data_changes", lambda *args: websocket.send(args[3]))
     ```
151. **Multi-Tenant RAG**: Tenant-specific RAG.
     ```python
     docs = session.execute(select(Document).where(Document.tenant_id == user.tenant_id).order_by(Document.embedding.l2_distance(vector)).limit(5)).scalars().all()
     ```
152. **Query Autocomplete**: Suggest queries.
     ```python
     suggestions = llm.run(f"Suggest SQLAlchemy queries for: {prompt}")
     ```
153. **Security Linting**: Detect dangerous queries.
     ```python
     if "DROP TABLE" in query:
         raise ValueError("Dangerous query detected")
     ```
154. **Vector + SQL Sync**: Sync with Redis.
     ```python
     results = session.execute(select(Document).where(Document.embedding == redis.get("vector"))).scalars().all()
     ```
155. **AI Dashboard Query**: Fetch live metrics.
     ```python
     results = session.execute(select(Metric).where(Metric.source == "live").order_by(desc(Metric.timestamp)).limit(10)).scalars().all()
     ```

## Conclusion
This `README.md` covers all 155 SQLAlchemy + PostgreSQL querysets, organized by category with complete examples. For full implementations, test data, and setup scripts, visit [https://github.com/xai/sqlalchemy-querysets](https://github.com/xai/sqlalchemy-querysets).
