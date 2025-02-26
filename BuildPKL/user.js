document.getElementById('submit_register').addEventListener('click', function(event) {
    event.preventDefault();
  
    const userName = document.getElementById('userName').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
  
    if (password !== confirmPassword) {
      alert('Mật khẩu và xác nhận mật khẩu không khớp.');
      return;
    }
  
    fetch('https://6659f760de346625136e95d5.mockapi.io/book_api/users', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ user: userName, password: password })
    })
    .then(response => response.json())
    .then(data => {
      console.log('Success:', data);
      alert('Đăng ký thành công!');
    })
    .catch((error) => {
      console.error('Error:', error);
      alert('Đăng ký thất bại!');
    });
});
  
// document.querySelector('#submit_login').addEventListener('click', function(event) {
//     const userName = document.getElementById('loginForm').querySelector('input[type="text"]').value;
//     const password = document.getElementById('loginForm').querySelector('input[type="password"]').value;
    
//     fetch('https://6659f760de346625136e95d5.mockapi.io/book_api/users')
//       .then(response => response.json())
//       .then(users => {
//         const user = users.find(user => user.user === userName && user.password === password);
//         if (user) {
//           alert('Đăng nhập thành công!');
//           document.querySelector('.modal').classList.remove('open');
//         } else {
//           alert('Thông tin đăng nhập không đúng.');
//         }
//       })
//       .catch(error => {
//         console.error('Error:', error);
//         alert('Đã xảy ra lỗi trong quá trình đăng nhập.');
//       });
// });

document.querySelector('#submit_login').addEventListener('click', function(event) {
  event.preventDefault();  // Ngăn chặn hành vi mặc định của nút submit
  
  const userName = document.getElementById('loginForm').querySelector('input[type="text"]').value;
  const password = document.getElementById('loginForm').querySelector('input[type="password"]').value;
  
  fetch('https://6659f760de346625136e95d5.mockapi.io/book_api/users')
    .then(response => response.json())
    .then(users => {
      const user = users.find(user => user.user === userName && user.password === password);
      if (user) {
        alert('Đăng nhập thành công!');
        document.querySelector('.modal').classList.remove('open');
        // Hiển thị tên người dùng trong khối "navbar-item navbar-item_user"
        document.querySelector('.navbar-item_user').textContent = user.user;
      } else {
        alert('Thông tin đăng nhập không đúng.');
      }
    })
    .catch(error => {
      console.error('Error:', error);
      alert('Đã xảy ra lỗi trong quá trình đăng nhập.');
    });
});