# MLP From Scratch — Neural Network with NumPy (No Frameworks)

پیاده‌سازی یک شبکه عصبی چندلایه (Multi-Layer Perceptron) از صفر با NumPy خالص، بدون استفاده از PyTorch یا TensorFlow، برای طبقه‌بندی ارقام دست‌نویس روی دیتاست MNIST.

## چرا بدون فریمورک؟

هدف این پروژه نشان دادن درک ریاضی پشت شبکه‌های عصبی است، نه صرفاً استفاده از یک API. Forward pass، backpropagation، محاسبه‌ی گرادیان و gradient descent همگی دستی و از پایه پیاده‌سازی شده‌اند — بدون `.backward()` خودکار.

## معماری

- ورودی: ۷۸۴ نورون (تصاویر ۲۸×۲۸ پیکسل)
- لایه‌ی مخفی: ۶۴ نورون با فعال‌سازی ReLU
- خروجی: ۱۰ نورون با فعال‌سازی Softmax (یک نورون برای هر رقم ۰ تا ۹)
- تابع خطا: Cross-Entropy Loss
- بهینه‌سازی: Gradient Descent ساده

## مقایسه

نتایج مدل با یک مدل کلاسیک (`LogisticRegression` از scikit-learn) روی همان داده مقایسه می‌شود، تا تفاوت عملکرد یک شبکه‌ی عصبی ساده در برابر یک مدل خطی مشخص شود.

## نحوه اجرا

```bash
pip install numpy scikit-learn matplotlib
python mlp_from_scratch.py
```

اجرای برنامه دیتاست MNIST را (در صورت نیاز) دانلود می‌کند، مدل را برای ۱۰۰ epoch آموزش می‌دهد، دقت روی داده‌ی تست را چاپ می‌کند، و نمودار کاهش loss را در فایل `loss_curve.png` ذخیره می‌کند.

## خروجی نمونه

```
دقت MLP از صفر: 0.9xxx
دقت Logistic Regression: 0.9xxx
```

## ساختار فایل

فایل به بخش‌های مجزا تقسیم شده: بارگذاری داده، مقداردهی اولیه وزن‌ها، توابع فعال‌سازی، forward pass، محاسبه‌ی loss، backpropagation، حلقه‌ی آموزش، ارزیابی، و رسم نمودار.
