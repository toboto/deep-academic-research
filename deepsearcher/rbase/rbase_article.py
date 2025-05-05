class RbaseAuthor:
    def __init__(self, name: str, ename: str = "", cname: str = "", is_corresponding: bool = False):
        self.name = name
        self.ename = ename
        self.cname = cname
        self.is_corresponding = is_corresponding

    def set_author_ids(self, author_ids: list[int]):
        self.author_ids = author_ids


class RbaseArticle:
    def __init__(self, article_data: dict = None, **kwargs):
        """
        初始化RbaseArticle对象

        参数:
            article_data: 包含文章数据的字典
            **kwargs: 其他关键字参数，可直接指定各个属性
        """
        if article_data is None:
            article_data = {}

        # 从字典或关键字参数中获取属性值
        self.article_id = (
            kwargs.get("article_id")
            or article_data.get("id")
            or article_data.get("raw_article_id", 0)
        )
        self.title = kwargs.get("title") or article_data.get("title", "")
        self.txt_file = kwargs.get("txt_file") or article_data.get("txt_file", "")
        self.authors = kwargs.get("authors") or article_data.get("authors", "")
        self.corresponding_authors = kwargs.get("corresponding_authors") or article_data.get(
            "corresponding_authors", ""
        )
        self.source_keywords = kwargs.get("source_keywords") or article_data.get(
            "source_keywords", ""
        )
        self.mesh_keywords = kwargs.get("mesh_keywords") or article_data.get("mesh_keywords", "")
        self.impact_factor = kwargs.get("impact_factor") or article_data.get("impact_factor", 0)
        self.pubdate = kwargs.get("pubdate") or article_data.get("pubdate", None)
        if self.pubdate is not None:
            # 将datetime.date类型转换为datetime.datetime，然后获取时间戳
            import datetime

            if isinstance(self.pubdate, datetime.date) and not isinstance(
                self.pubdate, datetime.datetime
            ):
                self.pubdate = datetime.datetime.combine(self.pubdate, datetime.time()).timestamp()
            else:
                # 如果已经是datetime.datetime类型，直接获取时间戳
                self.pubdate = self.pubdate.timestamp()
        else:
            self.pubdate = 0
        self.author_objects = []
        self.abstract = kwargs.get("abstract") or article_data.get("abstract", "")
        self.summary = kwargs.get("summary") or article_data.get("summary", self.abstract)
        self.journal_name = kwargs.get("journal_name") or article_data.get("journal_name", "")

    def set_author(self, author: RbaseAuthor):
        self.author_objects.append(author)
