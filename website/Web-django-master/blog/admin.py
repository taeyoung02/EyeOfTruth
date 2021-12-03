from django.contrib import admin





class CategoryAdmin(admin.ModelAdmin):
    prepopulated_fields = {'slug': ('name',)}
