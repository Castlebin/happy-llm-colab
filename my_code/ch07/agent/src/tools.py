from datetime import datetime
import wikipedia

# 获取当前时间和日期
def get_current_datetime() -> str:
    """
    获取当前日期和时间
    :return: 当前日期和时间的字符串表示，格式为 'YYYY-MM-DD HH:MM:SS'
    """
    current_datetime = datetime.now()
    formated_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return formated_datetime

# 统计字符串中某个字母出现的次数
def count_letter_in_string(input_string: str, letter: str) -> int:
    """
    统计字符串中某个字母出现的次数
    :param input_string: 输入的字符串
    :param letter: 要统计的字母
    :return: 字母在字符串中出现的次数
    """
    count = str(input_string).count(letter)
    return count

# 在维基百科中搜索指定查询的前 3 个页面摘要
def search_wikipedia(query: str) -> str:
    """
    在维基百科中搜索指定查询的前 3 个页面摘要
    :param query: 搜索查询字符串
    :return: 前 3 个页面的摘要
    """
    page_titles = wikipedia.search(query, results=3)
    summaries = []
    for title in page_titles:
        try:
            # 使用 page 方法获取页面内容
            wiki_page = wikipedia.page(title, auto_suggest=False)
            # 获取页面摘要
            summaries.append(f"页面: {title}\n摘要: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
        
    if not summaries:
        return "未找到相关的维基百科页面。"
    
    return "\n\n".join(summaries)

