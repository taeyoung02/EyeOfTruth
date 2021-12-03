from celery import Celery
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "my_site_prj.settings")

app = Celery('blog',
             broker='redis://localhost:6379',
             backend='redis://localhost:6379',
             include=['blog.views'],
             )

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()