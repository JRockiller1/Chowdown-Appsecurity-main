import DePayWidgets from '@depay/widgets'
var button = document.getElementById("pay");
   button.addEventListener("click",function(e){
    DePayWidgets.Payment({
  integration: 'e97d0ec9-2aca-47b4-a3b6-381dd818f07d',
  accept:[
    {
      blockchain: 'ethereum',
      amount: 0.00001,
      token: '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE',
      receiver: '0x51723E0581FCb99f5684F67E62975b1C47D15D21'
    }
  ]
,
    track: {
              method: (payment)=>axios.post('/track', payment)
                      }
})


    
  }); 