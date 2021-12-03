from django.test import TestCase, Client
from bs4 import BeautifulSoup
from .models import Post, Category, UploadFileModel
from django.utils import timezone
from django.contrib.auth.models import User
import pandas as pd


# TDD
def upload_post(file):
    blog_post = Post.objects.create(
        file=file
    )
    return blog_post

class test_ml(TestCase):
    def setUp(self):
        self.client = Client()  # client가 브라우저 역할 대신
        self.author_000 = User.objects.create(username='smith', password='no')

    def ml(self):
        post_000 = upload_post(
            file='kt_s.xlsx'
        )
        response = self.client.get('/upload/')  # Http://~~/blog의 소스를 가져오겠다
        self.assertEqual(response.status_code, 200)
        kt = pd.read_excel('_media/kt_s.xlsx')
        self.assertEqual(len(kt), 8001)

# def create_category(name='life', description=''):
#     category, is_created = Category.objects.get_or_create(
#         name=name,
#         description=description
#     )
#
#     category.slug = category.name.replace(' ', '-').replace('/', '')
#     category.save()
#
#     return category
#
#
# def create_post(title, content, author, category=None):
#     blog_post = Post.objects.create(
#         title=title,
#         content=content,
#         created=timezone.now(),
#         author=author,
#         category=category,
#     )
#     return blog_post
#
#
# def create_comment(post, text='a comment', author=None):
#     if author is None:
#         author, is_created = User.objects.get_or_create(
#             username='guest',
#             password='guest',
#         )
#
#     comment = Comment.objects.create(
#         post=post,
#         text=text,
#         author=author
#     )
#
#     return comment
#
#
# class TestModel(TestCase):
#     def setUp(self):
#         self.client = Client()  # client가 브라우저 역할 대신
#         self.author_000 = User.objects.create(username='smith', password='no')
#
#     def test_post(self):
#         category = create_category(
#
#         )
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000,
#             category=category,
#         )
#
#         self.assertEqual(category.post_set.count(), 1)  # Post를 post로 가져옴. category에서 post 불러옴
#
#     def test_comment(self):
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000
#         )
#
#         self.assertEqual(Comment.objects.count(), 0)
#
#         comment0 = create_comment(
#             post=post_000
#         )
#
#         comment1 = create_comment(
#             post=post_000,
#             text='second'
#         )
#
#         self.assertEqual(Comment.objects.count(), 2)
#         self.assertEqual(post_000.comment_set.count(), 2)
#
#
# class Testview(TestCase):
#     def setUp(self):
#         self.client = Client()  # client가 브라우저 역할 대신
#         self.author_000 = User.objects.create_user(username='smith', password='no')
#         self.user_obama = User.objects.create_user(username='obama', password='no')
#
#     def check_navbar(self, soup):
#         navbar = soup.find("div", id='navbar')
#         self.assertIn("Github", navbar.text)
#         self.assertIn("About me", navbar.text)
#
#     def check_right_side(self, soup):
#         # category card 에서
#         category_card = soup.find('div', id='category-card')
#         print(category_card)
#         # 미분류(1),정치/사회(1) 있어야함
#         self.assertIn('미분류 (1)', category_card.text)
#         self.assertIn('정치/사회 (1)', category_card.text)
#
#     def test_post_list_noPost(self):
#         response = self.client.get('/blog/')  # Http://~~/blog의 소스를 가져오겠다
#         self.assertEqual(response.status_code, 200)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         title = soup.title  # soup = html 태그 가져옴
#
#         self.assertIn('Blog', title.text)
#
#         self.check_navbar(soup)
#         # navbar = soup.find("div", id='navbar')
#         # self.assertIn("Blog", navbar.text)
#         # self.assertIn("About me", navbar.text)
#
#         self.assertEqual(Post.objects.count(), 0)
#         self.assertIn("아직 게시물이 없습니다", soup.body.text)
#
#     def test_post_list_Yespost(self):
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000,
#         )
#
#         post_001 = create_post(
#             title='The second post',
#             content='ss',
#             author=self.author_000,
#             category=create_category(name='정치/사회')
#         )
#
#         self.assertGreater(Post.objects.count(), 0)
#
#         response = self.client.get('/blog/')
#         self.assertEqual(response.status_code, 200)
#         soup = BeautifulSoup(response.content, 'html.parser')
#         body = soup.body
#         self.assertNotIn('아직 게시물이 없습니다.', body.text)
#         self.assertIn(post_000.title, body.text)
#
#         post_000_read = body.find('a', id='read-{}'.format(post_000.pk))
#         self.assertEqual(post_000_read['href'], post_000.get_absolute_url())
#
#         self.check_right_side(soup)
#
#         # main-div 포스트에는 정치/사회
#         main = soup.find('div', id='main-div')
#         self.assertIn('정치/사회', main.text)
#
#     def test_post_detail(self):
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000,
#         )
#
#         comment0 = create_comment(post_000, text='a test comment', author=self.user_obama)
#
#         post_001 = create_post(
#             title='The second post',
#             content='ss',
#             author=self.author_000,
#             category=create_category(name='정치/사회')
#         )
#
#         self.assertGreater(Post.objects.count(), 0)
#         post_000_url = post_000.get_absolute_url()
#         self.assertEqual(post_000_url, '/blog/{}/'.format(post_000.pk))
#         # post의 self와 testview의 self가 다름
#         response = self.client.get(post_000_url)  # post_000.get_absolute_url()의 소스를 가져오겠다
#         self.assertEqual(response.status_code, 200)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         title = soup.title  # soup = html 태그 가져옴
#
#         self.assertEqual(title.text, '{} - Blog'.format(post_000.title))
#
#         self.check_navbar(soup)
#
#         body = soup.body
#
#         main = body.find('div', id='main-div')
#         self.assertIn(post_000.title, main.text)
#         self.assertIn(post_000.author.username, main.text)
#
#         self.assertIn(post_000.content, main.text)
#
#         self.check_right_side(soup)
#
#         # Comment
#         comments_div = main.find('div', id='comment-list')
#         self.assertIn(comment0.author.username, comments_div.text)
#         self.assertIn(comment0.text, comments_div.text)
#
#         # login 한 경우
#         # post author == login사용자 이면 edit버튼 있다
#         login_sucess = self.client.login(username='smith', password='no')
#         self.assertTrue(login_sucess)
#
#         response = self.client.get(post_000_url)  # post_000.get_absolute_url()의 소스를 가져오겠다
#         self.assertEqual(response.status_code, 200)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         main = soup.find('div', id='main-div')
#
#         self.assertEqual(post_000.author, self.author_000)
#         self.assertIn('EDIT', main.text)
#
#     def test_post_list_byCategory(self):
#         category_politics = create_category(name='정치/사회')
#
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000,
#         )
#
#         post_001 = create_post(
#             title='The second post',
#             content='ss',
#             author=self.author_000,
#             category=category_politics
#         )
#
#         response = self.client.get(category_politics.get_absolute_url())
#         self.assertEqual(response.status_code, 200)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         # self.assertEqual('Blog - {}'.format(category_politics.name), soup.title.text)
#
#         main = soup.find('div', id='main-div')
#         self.assertNotIn('미분류', main.text)
#         self.assertIn(category_politics.name, main.text)
#
#     def test_post_create(self):
#         response = self.client.get('/blog/create/')
#         self.assertNotEqual(response.status_code, 200)
#
#         self.client.login(username='smith', password='no')
#         response = self.client.get('/blog/create/')
#         self.assertEqual(response.status_code, 200)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         main = soup.find('div', id='main-div')
#
#     def test_post_update(self):
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000,
#         )
#
#         self.assertEqual(post_000.get_update_url(), post_000.get_absolute_url() + 'update/')
#
#         response = self.client.get(post_000.get_update_url())
#         self.assertEqual(response.status_code, 200)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         main = soup.find('div', id='main-div')
#
#         # edit시 author 못바꾸게
#         self.assertNotIn('Created', main.text)
#         self.assertNotIn('Author', main.text)
#
#     def test_new_comment(self):
#         post_000 = create_post(
#             title='The first post',
#             content='hello wolrd.',
#             author=self.author_000,
#         )
#
#         response = self.client.post(
#             post_000.get_absolute_url() + 'new_comment',
#             {'text': 'A test comment for the first'},
#             follow=True
#         )
#         self.assertEqual(response.status_code, 200)  # urls.py에 구현이 안되었기때문에 404를 받음
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         main = soup.find('div', id='main-div')
#
#         self.assertIn(post_000.title, main.text)
#         self.assertIn('A test comment', main.text)
#
#     def test_pagination(self):
#         # pager가 작은경우
#         for i in range(0, 3):
#             post_000 = create_post(
#                 title='The post {}'.format(i),
#                 content='{}'.format(i),
#                 author=self.author_000,
#             )
#
#         response = self.client.get('/blog/')
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         main = soup.find('div', id='main-div')
#
#         self.assertNotIn('Older', soup.body.text)
#         self.assertNotIn('Newer', soup.body.text)
#
#         for i in range(3, 10):
#             post_000 = create_post(
#                 title='The post {}'.format(i),
#                 content='{}'.format(i),
#                 author=self.author_000,
#             )
#
#         response = self.client.get('/blog/')
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#         main = soup.find('div', id='main-div')
#
#     def test_search(self):
#         post_000 = create_post(
#             title='stay Fool, stay hungry',
#             content='hello wolrd.',
#             author=self.author_000,
#         )
#
#         post_001 = create_post(
#             title='The',
#             content='hello.',
#             author=self.author_000,
#         )
#
#         response = self.client.get('/blog/search/stay Fool/')
#         self.assertEqual(response.status_code, 200)
#         soup = BeautifulSoup(response.content, 'html.parser')
#         self.assertIn(post_000.title, soup.body.text)
#         self.assertNotIn(post_001.title, soup.body.text)
#
#         response = self.client.get('/blog/search/The/')
#         self.assertEqual(response.status_code, 200)
#         soup = BeautifulSoup(response.content, 'html.parser')
#         self.assertIn(post_001.title, soup.body.text)
#         self.assertNotIn(post_000.title, soup.body.text)
#
#     # venv/Scripts/activate
