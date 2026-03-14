content = open('main.py').read()

# Fix 1: retrieve more chunks with lower threshold
old = '''def retrieve_context(query, top_k=4):
    if not chunks or is_greeting(query):
        return ""
    query_embedding = np.array(get_embedding(query))
    embeddings_array = np.array(embeddings)
    similarities = np.dot(embeddings_array, query_embedding) / (
        np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context_parts = [chunks[i] for i in top_indices if similarities[i] > 0.3]
    return "\\n\\n---\\n\\n".join(context_parts)'''

new = '''def retrieve_context(query, top_k=6):
    if not chunks or is_greeting(query):
        return ""
    query_embedding = np.array(get_embedding(query))
    embeddings_array = np.array(embeddings)
    similarities = np.dot(embeddings_array, query_embedding) / (
        np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context_parts = [chunks[i] for i in top_indices if similarities[i] > 0.2]
    return "\\n\\n---\\n\\n".join(context_parts)'''

# Fix 2: better system prompt forcing detail
old2 = '''SYSTEM_PROMPT_AR = """أنتِ مساعدة ذكية لنادي IEEE الطلابي في جامعة الملك عبدالعزيز (قسم الطالبات).
تحدثي باللهجة السعودية البسيطة والودودة.
اجعلي ردودك مفيدة ومفصلة (4-6 جمل).
أجيبي فقط عن أسئلة متعلقة بنادي IEEE وجامعة الملك عبدالعزيز.
إذا كان السؤال تحية، ردي بترحيب واسألي كيف تقدرين تساعدين.
إذا وجد في السياق رابط أو نموذج تسجيل، يجب ذكره كاملاً في إجابتك دون حذفه.
لا تذكري روابط nitter.net أبداً.
إذا لم تعرفي الإجابة قولي: للأسف ما عندي هالمعلومة حالياً، تقدري تتواصلي معنا على kau.ieee.sb@gmail.com"""'''

new2 = '''SYSTEM_PROMPT_AR = """أنتِ مساعدة ذكية لنادي IEEE الطلابي في جامعة الملك عبدالعزيز (قسم الطالبات).
تحدثي باللهجة السعودية البسيطة والودودة.

قواعد الإجابة:
- اجعلي ردودك مفصلة وشاملة - استخدمي كل المعلومات المتاحة في السياق
- إذا سألوا عن اللجان، اذكري كل لجنة ومهامها بالتفصيل
- إذا سألوا عن الانضمام، اذكري الرابط كاملاً والخطوات
- إذا وجد رابط في السياق يجب ذكره كاملاً دون حذف
- لا تذكري روابط nitter.net أبداً
- أجيبي فقط عن أسئلة متعلقة بنادي IEEE
- إذا كان السؤال تحية، ردي بترحيب قصير واسألي كيف تقدرين تساعدين
- إذا لم تعرفي الإجابة: للأسف ما عندي هالمعلومة، تقدري تتواصلي على kau.ieee.sb@gmail.com"""'''

content = content.replace(old, new)
content = content.replace(old2, new2)
open('main.py', 'w').write(content)
print('done')
